# Reproducing the dataset

The on-chain CSVs this project trains on are **not included in the repository**. They come from [bitcoin-data.com](https://bitcoin-data.com) (BGeometrics), whose [terms of service](https://charts.bgeometrics.com/terms.html) do not permit redistributing downloaded data. To retrain the model (`model.py`) or run the leakage demo (`leakage_demo.py`) you need to download them yourself into this folder. The backtest (`backtest.py`) does not need them it runs from the predictions CSV included in `results/`.

## How to download

Every metric below has a page on [bitcoin-data.com](https://bitcoin-data.com) with a **free CSV download** (no account needed). That is exactly how this dataset was built: 24 manual downloads, one per metric. It is tedious, budget 20–30 minutes and be patient with the site. The programmatic API was not used because it requires a paid plan; the free path is the manual per-metric download.

Save each file in this folder (`assets/`) under the exact filename in the table, `data.py` looks them up by name.

## The 24 files

Downloaded **September 25–26, 2025**. The date ranges below are what the study used; fresher downloads will simply extend them.

| File (exact name) | Metric | Column(s) | Range used |
|---|---|---|---|
| `btc-ohlc.csv` | **BTC daily OHLCV (the main series)** | `open, high, low, close, volume` | **2014-12-31 → 2025-09-26** |
| `aviv.csv` | AVIV ratio | `aviv` | 2010-10-22 → 2025-09-24 |
| `bitcoin-dominance.csv` | Bitcoin dominance | `bitcoinDominance` | 2013-04-29 → 2025-09-25 |
| `cdd.csv` | Coin Days Destroyed | `cdd` | 2010-08-17 → 2025-09-25 |
| `difficulty-btc.csv` | Mining difficulty | `difficultyBtc` | 2011-01-01 → 2025-09-24 |
| `hashprice.csv` | Hashprice | `hashprice` | 2011-01-01 → 2025-09-24 |
| `hashrate.csv` | Hashrate | `hashrate` | 2011-01-01 → 2025-09-24 |
| `hashribbons.csv` | Hash Ribbons signal | `hashribbons` (`Up`/`Down`, mapped to 1/0) | 2012-01-01 → 2025-09-24 |
| `hodl-waves-supply.csv` | HODL waves (supply by age band) | `0d_1d, 1d_1w, 1w_1m, 1m_3m, 3m_6m, 6m_1y, 1y_2y, 2y_3y, 3y_4y, 4y_5y, 5y_7y, 7y_10y, 10y_` | 2011-01-01 → 2025-09-24 |
| `miner-balances.csv` | Miner balances | `minerBalances` | 2012-05-09 → 2025-09-25 |
| `mvrv.csv` | MVRV ratio | `mvrv` | 2012-01-01 → 2025-09-25 |
| `mvrv-zscore.csv` | MVRV Z-score | `mvrv-zscore` | 2009-01-03 → 2025-09-25 |
| `nrpl-btc.csv` | Net Realized Profit/Loss | `nrplBtc` | 2010-07-18 → 2025-09-25 |
| `nupl.csv` | Net Unrealized Profit/Loss | `nupl` | 2013-01-01 → 2025-09-25 |
| `nvts.csv` | NVT Signal | `nvts` | 2012-03-30 → 2025-09-25 |
| `puell-multiple.csv` | Puell Multiple | `puellMultiple` | 2012-05-09 → 2025-09-25 |
| `realized-price.csv` | Realized price | `realizedPrice` | 2009-01-03 → 2025-09-25 |
| `reserve-risk.csv` | Reserve Risk | `reserveRisk` | 2010-08-17 → 2025-09-25 |
| `sopr.csv` | SOPR | `sopr` | 2010-07-17 → 2025-09-25 |
| `supply-current.csv` | Circulating supply | `supplyCurrent` | 2009-01-03 → 2025-09-25 |
| `terminal-price.csv` | Terminal price | `terminalPrice` | 2010-08-17 → 2025-09-24 |
| `thermo-cap.csv` | Thermocap | `thermoCap` | 2012-05-09 → 2025-09-25 |
| `thermo-price.csv` | Thermo price | `thermoPrice` | 2012-05-09 → 2025-09-25 |
| `true-market-mean.csv` | True Market Mean | `trueMarketMean` | 2010-10-22 → 2025-09-25 |

Expected CSV shape (what the site's download button produces):

```
d,unixTs,<metric>
2012-01-01,1325376000,1.0759757060007067
...
```

`data.py` parses the first column as the date index and drops `unixTs`.

## What the pipeline actually uses

`data.py` joins everything onto the OHLCV index, forward-fills gaps (never backfills, that would copy the future backwards), then drops the first 1,470 rows (sparse early history) and the trailing partial rows. The **effective dataset is 2019-01-09 → 2025-09-23** (~2,450 daily rows); the first 90 of those are consumed as warm-up by the causal rolling z-score normalization.

The macro features (gold, oil, S&P 500, VIX, treasury futures, …) are not manual downloads. `data.py` fetches them automatically from Yahoo Finance via `yfinance` over the same date range. You only need the 24 files above.
