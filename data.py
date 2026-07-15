"""Build the BTC feature dataset: OHLC + on-chain metrics + indicators + macro tickers."""

from pathlib import Path

import pandas as pd
import ta
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "assets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PREDICTIONS_CSV = RESULTS_DIR / "best_test_dataset.csv"  # written by model.py, read by backtest.py

ONCHAIN_CSVS = [
    "hodl-waves-supply",
    "aviv",
    "bitcoin-dominance",
    "supply-current",
    "terminal-price",
    "cdd",
    "difficulty-btc",
    "hashprice",
    "hashrate",
    "hashribbons",
    "miner-balances",
    "mvrv",
    "mvrv-zscore",
    "nrpl-btc",
    "nupl",
    "nvts",
    "puell-multiple",
    "realized-price",
    "reserve-risk",
    "sopr",
    "thermo-cap",
    "thermo-price",
    "true-market-mean",
]

YF_TICKERS = [
    "GC=F",
    "CL=F",
    "ZB=F",
    "ZN=F",
    "ZT=F",
    "SI=F",
    "^GSPC",
    "^DJI",
    "^IXIC",
    "^NYA",
    "^XAX",
    "^VIX",
    "^TNX",
]


def _read_csv(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0)
    df.index = pd.to_datetime(df.index).date
    return df.drop(columns="unixTs")


def add_indicators(df):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    for w in (5, 20, 50):
        df[f"SMA_{w}"] = ta.trend.sma_indicator(close, window=w)
    for w in (12, 26):
        df[f"EMA_{w}"] = ta.trend.ema_indicator(close, window=w)
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD_12_26_9"] = macd.macd()
    df["MACDh_12_26_9"] = macd.macd_diff()
    df["MACDs_12_26_9"] = macd.macd_signal()
    df["atr_30"] = ta.volatility.average_true_range(high, low, close, window=30)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["Daily_Return"] = close.pct_change()
    df["OBV"] = ta.volume.on_balance_volume(close, vol)
    df["VWAP"] = ta.volume.volume_weighted_average_price(high, low, close, vol, window=14)
    df["volume_relative"] = vol / vol.rolling(20).mean()
    return df


def load_dataset(skip_rows=1470):
    """Joined dataset with a next-day-direction Target column."""
    df = _read_csv("btc-ohlc")
    df.index.name = "Date"
    for name in ONCHAIN_CSVS:
        other = _read_csv(name)
        if name == "hashribbons":
            other["hashribbons"] = other["hashribbons"].map({"Up": 1, "Down": 0})
        df = df.join(other, how="left")

    df = add_indicators(df)

    macro = yf.download(" ".join(YF_TICKERS), start=df.index[0], end=df.index[-1])["Close"]
    df = df.join(macro, how="left")

    # ffill only: backfilling would copy future values backwards
    df = df.ffill()

    # drop early history with sparse metrics and the last 2 partial rows
    df = df.iloc[skip_rows:-2].copy()
    df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    # the final row has no next close, so its label would be fabricated
    return df.iloc[:-1]
