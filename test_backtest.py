"""
Self-check for strategy logic.
Run: python test_backtest.py
"""

import numpy as np
import pandas as pd

from backtest import (
    bollinger_strategy,
    evaluate,
    hodl_strategy,
    predicted_target_strategy,
)
from model import create_sequences

# sequence alignment: label = window's LAST row's target, features end on the labeled day
xs, ys = create_sequences(np.arange(10).reshape(-1, 1), np.arange(10), 4)
assert len(xs) == 7 and ys[0] == 3 and xs[0][-1, 0] == 3


def _df(close, **cols):
    idx = pd.date_range("2024-01-01", periods=len(close))
    return pd.DataFrame({"close": close, **cols}, index=idx)


# long: dip below lower band, take profit at the middle band
df = _df([100, 90, 100, 110], bb_low=[95] * 4, bb_mid=[105] * 4, bb_high=[200] * 4)
trades = bollinger_strategy(df, stop_loss=0.5, balance=1000)
assert len(trades) == 1 and trades["type"][0] == "TP_LONG" and trades["profit"][0] > 0

# longs disabled → same dip produces no trade
assert bollinger_strategy(df, stop_loss=0.5, balance=1000, longs=False).empty

# a short on a falling price must profit
df = _df([100, 100, 80, 80, 120], Predicted_Target=[0, 0, 0, 1, 1])
trades = predicted_target_strategy(df, balance=1000)
assert trades["type"][0] == "SHORT" and trades["profit"][0] > 0, trades.iloc[0]

# fees: 10 shares shorted at 100, covered at 80 → 0.1% per side costs 10*(100+80)*0.001 = 1.8
gross = predicted_target_strategy(df, balance=1000, fee=0)
assert np.isclose(gross["profit"][0] - trades["profit"][0], 1.8)

# hodl compounds exactly with price (fee disabled for exactness)
df = _df([100, 110, 121])
trades = hodl_strategy(df, balance=1000, fee=0)
assert np.isclose(trades["balance"].iloc[-1], 1210)

metrics = evaluate(trades, balance=1000)
assert np.isclose(metrics["final_balance"], 1210) and metrics["win_rate_%"] == 100

# mark-to-market: a long that dips mid-trade shows the dip in drawdown
idx = pd.date_range("2024-01-01", periods=3)
closes = pd.Series([100.0, 80.0, 120.0], index=idx)
trade = {"type": "LONG", "entry": 100.0, "exit": 120.0, "profit": 200.0, "balance": 1200.0}
trades = pd.DataFrame([{**trade, "entry_time": idx[0], "exit_time": idx[2]}])
metrics = evaluate(trades, balance=1000, closes=closes)
assert np.isclose(metrics["max_drawdown_%"], -20)  # 1000 -> 800 -> 1200


print("ok")
