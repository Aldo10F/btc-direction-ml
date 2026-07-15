"""
Strategies + performance metrics.
Run: python backtest.py
"""

import pandas as pd

FEE = 0.001  # taker fee per side, as a fraction of traded notional


def _trade(kind, entry, exit_, profit, balance, entry_time, exit_time):
    return {
        "type": kind,
        "entry": entry,
        "exit": exit_,
        "profit": profit,
        "balance": balance,
        "entry_time": entry_time,
        "exit_time": exit_time,
    }


def bollinger_strategy(df, stop_loss=0.03, balance=1000.0, longs=True, shorts=True, fee=FEE):
    """Bollinger Bands strategy: long below the lower band, short above the upper band, exit at the
    middle band (TP) or at stop_loss. `fee` is charged on entry and exit notional."""
    position, trades = None, []
    entry_price, entry_time, shares = 0.0, None, 0.0
    for i in range(1, len(df)):
        price, when = df["close"].iloc[i], df.index[i]
        if position is None:
            if longs and price < df["bb_low"].iloc[i]:
                position, entry_price, entry_time = "long", price, when
                shares = balance / price
            elif shorts and price > df["bb_high"].iloc[i]:
                position, entry_price, entry_time = "short", price, when
                shares = balance / price
            continue
        mid = df["bb_mid"].iloc[i]
        if position == "long":
            tp = price >= mid
            sl = price <= entry_price * (1 - stop_loss)
            pnl = shares * (price - entry_price)
        else:
            tp = price <= mid
            sl = price >= entry_price * (1 + stop_loss)
            pnl = shares * (entry_price - price)
        if tp or sl:
            pnl -= fee * shares * (entry_price + price)
            balance += pnl
            trades.append(
                _trade(
                    ("TP_" if tp else "SL_") + position.upper(),
                    entry_price,
                    price,
                    pnl,
                    balance,
                    entry_time,
                    when,
                )
            )
            position = None
    return pd.DataFrame(trades)


def predicted_target_strategy(df, balance=1000.0, fee=FEE):
    """Always in the market, flipping long/short on the model's direction signal.
    The final open position is closed at the last bar. `fee` is charged on entry and exit notional
    of every round trip."""
    position, trades = None, []
    entry_price, entry_time, shares = 0.0, None, 0.0
    for i in range(1, len(df)):
        price, when = df["close"].iloc[i], df.index[i]
        wanted = "long" if df["Predicted_Target"].iloc[i] == 1 else "short"
        if position == wanted:
            continue
        if position is not None:
            pnl = shares * ((price - entry_price) if position == "long" else (entry_price - price))
            pnl -= fee * shares * (entry_price + price)
            balance += pnl
            trades.append(
                _trade(position.upper(), entry_price, price, pnl, balance, entry_time, when)
            )
        position, entry_price, entry_time = wanted, price, when
        shares = balance / price
    if position is not None:
        price, when = df["close"].iloc[-1], df.index[-1]
        pnl = shares * ((price - entry_price) if position == "long" else (entry_price - price))
        pnl -= fee * shares * (entry_price + price)
        trades.append(
            _trade(
                position.upper(),
                entry_price,
                price,
                pnl,
                balance + pnl,
                entry_time,
                when,
            )
        )
    return pd.DataFrame(trades)


def hodl_strategy(df, balance=1000.0, fee=FEE):
    """Buy and hold, booked as daily close-to-close trades. `fee` is charged once, on the entry."""
    equity = (balance * (1 - fee) / df["close"].iloc[0]) * df["close"]
    return pd.DataFrame(
        {
            "type": "DAILY_HODL",
            "entry": df["close"].values[:-1],
            "exit": df["close"].values[1:],
            "profit": equity.diff().dropna().values,
            "balance": equity.values[1:],
            "entry_time": df.index[:-1],
            "exit_time": df.index[1:],
        }
    )


def _daily_equity(trades, closes, balance=1000.0):
    """Daily mark-to-market equity: open positions are valued at every close, so volatility and
    drawdown during a trade are visible, not just the booked balance at exit."""
    closes = closes.copy()
    closes.index = pd.to_datetime(closes.index)
    equity = pd.Series(float(balance), index=closes.index)
    prev = balance
    for _, t in trades.iterrows():
        entry, exit_ = pd.to_datetime(t["entry_time"]), pd.to_datetime(t["exit_time"])
        sign = -1 if "SHORT" in t["type"] else 1
        shares = prev / t["entry"]
        open_days = (equity.index > entry) & (equity.index < exit_)
        equity[open_days] = prev + sign * shares * (closes[open_days] - t["entry"])
        equity[equity.index >= exit_] = t["balance"]
        prev = t["balance"]
    return equity


def evaluate(trades, balance=1000.0, risk_free_rate=0.04, closes=None):
    """Print and return performance metrics (annualized over 365 days). Pass `closes` (daily close
    series covering the trade window) to mark open positions to market; without it, equity is only
    sampled at trade exits, which understates volatility and drawdown."""
    if trades.empty:
        print("No trades executed.")
        return {}
    trades = trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    start = pd.to_datetime(trades["entry_time"]).min()
    if closes is not None:
        daily = _daily_equity(trades, closes, balance).resample("D").ffill()
        daily = daily[daily.index >= start]
    else:
        daily = trades.set_index("exit_time")["balance"].resample("D").ffill()
    returns = daily.pct_change().dropna()
    years = ((daily.index[-1] - start).days + 1) / 365.25
    final = trades["balance"].iloc[-1]

    excess = returns - risk_free_rate / 365
    std, downside = returns.std(), returns[returns < 0].std()
    drawdown = (daily / daily.cummax() - 1).min()
    wins = trades[trades["profit"] > 0]
    losses = trades[trades["profit"] < 0]
    cagr = (final / balance) ** (1 / years) - 1 if years > 0 and final > 0 else 0.0

    metrics = {
        "trades": len(trades),
        "win_rate_%": len(wins) / len(trades) * 100,
        "total_pnl": trades["profit"].sum(),
        "final_balance": final,
        "return_%": (final / balance - 1) * 100,
        "cagr_%": cagr * 100,
        "sharpe": excess.mean() / std * 365**0.5 if std > 0 else 0.0,
        "sortino": (
            excess.mean() / downside * 365**0.5 if pd.notna(downside) and downside > 0 else 0.0
        ),
        "max_drawdown_%": drawdown * 100,
        "profit_factor": (
            wins["profit"].sum() / -losses["profit"].sum() if len(losses) else float("inf")
        ),
        "calmar": -cagr / drawdown if drawdown < 0 else float("inf"),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:,.2f}")
    return metrics


def save_equity_curve(trades, title, path, closes=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if closes is not None:
        daily = _daily_equity(trades, closes)
    else:
        daily = trades.copy()
        daily["exit_time"] = pd.to_datetime(daily["exit_time"])
        daily = daily.set_index("exit_time")["balance"].resample("D").ffill()
    ax = daily.plot(title=f"{title} — equity curve", figsize=(10, 5), grid=True)
    ax.set_ylabel("Balance (USD)")
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def save_trades_chart(df, trades, title, path, bands=True):
    """Candlestick chart with entry/exit markers. Bollinger bands optional."""
    import mplfinance as mpf

    dfp = df[["open", "high", "low", "close", "volume", "bb_high", "bb_low", "bb_mid"]].copy()
    dfp.index = pd.to_datetime(dfp.index)

    markers = {"entry_long": [], "entry_short": [], "exit": []}
    for _, t in trades.iterrows():
        side = "entry_short" if "SHORT" in t["type"] else "entry_long"
        markers[side].append(pd.to_datetime(t["entry_time"]))
        markers["exit"].append(pd.to_datetime(t["exit_time"]))

    plots = (
        [
            mpf.make_addplot(dfp["bb_high"], color="blue", linestyle="--", label="BB Upper"),
            mpf.make_addplot(dfp["bb_low"], color="blue", linestyle="--", label="BB Lower"),
            mpf.make_addplot(dfp["bb_mid"], color="purple", label="BB Middle"),
        ]
        if bands
        else []
    )
    for key, marker, color, offset in [
        ("entry_long", "^", "lime", 0.99),
        ("entry_short", "v", "red", 1.01),
        ("exit", "x", "orange", 1.0),
    ]:
        series = pd.Series(float("nan"), index=dfp.index)
        times = [t for t in markers[key] if t in series.index]
        if not times:
            continue
        series.loc[times] = dfp.loc[times, "close"] * offset
        plots.append(
            mpf.make_addplot(
                series,
                type="scatter",
                markersize=80,
                marker=marker,
                color=color,
                label=key,
            )
        )

    mpf.plot(
        dfp,
        type="candle",
        style="yahoo",
        title=title,
        ylabel="Price",
        addplot=plots,
        figsize=(14, 8),
        show_nontrading=False,
        savefig={"fname": path, "dpi": 150, "bbox_inches": "tight"},
    )


if __name__ == "__main__":
    from data import PREDICTIONS_CSV, RESULTS_DIR

    # the predictions CSV carries OHLCV + bands (saved by model.py), so no dataset rebuild needed
    df = pd.read_csv(PREDICTIONS_CSV, index_col=0)
    df.index = pd.to_datetime(df.index).date

    all_metrics = {}
    for name, trades in [
        ("bollinger_full", bollinger_strategy(df)),
        ("bollinger_bullish", bollinger_strategy(df, shorts=False)),
        ("bollinger_bearish", bollinger_strategy(df, longs=False)),
        ("model_signal", predicted_target_strategy(df)),
        ("hodl", hodl_strategy(df)),
    ]:
        print(f"\n=== {name} ===")
        all_metrics[name] = evaluate(trades, closes=df["close"])
        if not trades.empty:
            save_equity_curve(trades, name, RESULTS_DIR / f"equity_{name}.png", closes=df["close"])
            if name != "hodl":
                save_trades_chart(
                    df,
                    trades,
                    name,
                    RESULTS_DIR / f"trades_{name}.png",
                    bands=name.startswith("bollinger"),
                )

    pd.DataFrame(all_metrics).round(2).to_csv(RESULTS_DIR / "metrics.csv")
    print(f"\nSaved metrics.csv and charts to {RESULTS_DIR}")
