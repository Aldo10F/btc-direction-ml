"""
Self-check for strategy logic, plus a regression section that pins every number
published in README.md to the shipped results/ artifacts. If a number in the
README and its artifact drift apart, this fails.
Run: python test_backtest.py
"""

import json

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


# ---------------------------------------------------------------------------
# README regression: every published number must match its results/ artifact.
# ---------------------------------------------------------------------------
from data import PREDICTIONS_CSV, RESULTS_DIR  # noqa: E402

# replication table <- results/leakage_demo.csv (the artifact of record: the
# on-chain inputs can't be redistributed, so the demo can't rerun everywhere)
demo = pd.read_csv(RESULTS_DIR / "leakage_demo.csv", index_col=0)["test_accuracy"]
assert [round(v * 100, 1) for v in demo] == [52.0, 54.0, 50.9, 53.4, 97.5], demo

# search-winner stats <- results/best_params.json
params = json.loads((RESULTS_DIR / "best_params.json").read_text())
assert round(params["val_sharpe"], 2) == 3.27
assert round(params["test_sharpe"], 2) == 1.29
assert round(params["val_accuracy"] * 100, 1) == 56.1
assert round(params["test_accuracy"] * 100, 1) == 52.2

# backtest table <- results/metrics.csv, README rows in metrics.csv columns
m = pd.read_csv(RESULTS_DIR / "metrics.csv", index_col=0)
readme_table = {  # return_%, cagr_%, sharpe, max_drawdown_%, win_rate_%, trades
    "model_signal": (99, 79, 1.44, -28, 100, 2),
    "hodl": (75, 60, 1.17, -28, 51, None),
    "bollinger_full": (71, 60, 1.69, -12, 65, 23),
    "bollinger_bullish": (53, 45, 1.60, -7, 80, 10),
    "bollinger_bearish": (12, 11, 0.46, -12, 54, 13),
}
for name, (ret, cagr, sharpe, mdd, win, n_trades) in readme_table.items():
    col = m[name]
    got = (round(col["return_%"]), round(col["cagr_%"]), col["sharpe"],
           round(col["max_drawdown_%"]), round(col["win_rate_%"]))
    assert got == (ret, cagr, sharpe, mdd, win), (name, got)
    assert n_trades is None or col["trades"] == n_trades, name

# the prose claims <- results/best_test_dataset.csv
p = pd.read_csv(PREDICTIONS_CSV, index_col=0)
assert len(p) == 433 and (p.index[0], p.index[-1]) == ("2024-07-18", "2025-09-23")
assert round(p["Target"].mean() * 100, 1) == 51.5  # always-long base rate
assert round((p["Target"] == p["Predicted_Target"]).mean() * 100, 1) == 52.2
assert (p["Predicted_Target"] == 1).sum() == 410  # "410 of 433 predictions are up"
# one short->long flip on 2024-08-10, never trades again
assert list(p.index[p["Predicted_Target"].diff().fillna(0) != 0]) == ["2024-08-10"]

# robustness section <- results/robustness.csv + results/robustness.json
r = pd.read_csv(RESULTS_DIR / "robustness.csv", index_col=0)
assert (round(r["model_return_%"].iloc[0], 1), round(r["hodl_return_%"].iloc[0], 1)) == (19.1, 4.7)
# in every later slice the model is long every day: accuracy collapses to the base rate
assert (r["accuracy_%"].iloc[1:] == r["base_rate_%"].iloc[1:]).all()
rj = json.loads((RESULTS_DIR / "robustness.json").read_text())
assert rj["paths"] == 434 and rj["paths_beating_always_long"] == 80
assert rj["published_flip_date"] == "2024-08-10" and rj["published_percentile"] == 89.9
assert round(rj["median_final_balance"]) == 653 and round(rj["always_long_final_balance"]) == 1677
assert round(rj["published_final_balance"], 2) == 1989.96  # backtest table's model_signal
b = rj["binomial"]
assert (b["correct"], b["days"], b["base_rate_%"], b["p_value"]) == (226, 433, 51.5, 0.405)

# random-baselines bullet <- results/robustness.json (all with-fees figures)
rb = rj["random_baseline"]
assert (rb["sims"], rb["seed"], round(rb["hodl_final_balance"])) == (1000, 42, 1749)
cf, bm = rb["coin_flip"]["fees"], rb["bias_matched"]["fees"]
assert rb["bias_matched"]["p_long"] == 0.947  # "the model's own 95% long bias"
assert (round(cf["median_final_balance"]), cf["pct_positive_pnl"]) == (608, 14.0)  # "86% lose"
assert cf["model_percentile"] == 99.3
assert (round(bm["median_final_balance"]), bm["pct_positive_pnl"]) == (1450, 95.2)
assert (bm["pct_beating_model"], bm["model_percentile"]) == (6.3, 93.7)  # "63 of 1,000"


print("ok")
