# BTC Direction Prediction: A Replication Study

Papers keep reporting 70%+ accuracy (some as high as 90%) predicting whether Bitcoin closes up or down tomorrow, using CNNs/LSTMs over technical indicators and on-chain metrics. This project sets out to debunk those claims on their own terms: same target, same feature families, same model class the papers use. Rebuilt with an evaluation designed so that leaking, peeking, or cherry-picking is structurally impossible. It finds what an efficient market says it should find:

**~52% test accuracy, statistically indistinguishable from the 51.5% always-long base rate and a "winning" trading signal that is just buy-and-hold plus one lucky flip.**

A companion experiment (`leakage_demo.py`) then shows how the headline numbers get manufactured and it isn't the leak that usually gets blamed. The full mathematical specification of the pipeline, every formula from raw CSV to reported Sharpe, and why each design choice closes a specific loophole is in [MODEL.md](MODEL.md).

## The replication experiment

Same data, same features, same models, only the evaluation bugs differ:

| Arm | Test accuracy |
|-----|--------------:|
| Honest CNN (causal z-scores, chronological split) | 52.9% |
| Honest LightGBM baseline | 54.1% |
| Bug A: CNN, scaler fit on the full dataset + shuffled split | 48.9% |
| Bug B: CNN, label's move already inside the feature window | 53.5% |
| **Bug B: LightGBM, same misaligned label** | **97.5%** |

Two things fell out of this that I didn't expect:

- **The usually-blamed leaks barely matter for direction targets** (bug A). Daily up/down labels are near-independent coin flips, so fitting the scaler on all data and shuffling overlapping windows into train/test has almost no signal to leak.
- **Target misalignment is the real accuracy machine** (bug B). Label a sample with a price move its features already contain a one-line indexing bug when building sequences  and a tabular model scores 97.5% instantly. The Conv1D is largely immune only because `GlobalAveragePooling1D` is position-blind and can't isolate a leak parked in the window's last row. Many published pipelines are one `shift(-1)` away from this bug, and 65–90% claims are consistent with partial versions of it (features that encode the labeled move indirectly).

Reproduce with `python leakage_demo.py` (→ `results/leakage_demo.csv`).

## The honest model, honestly evaluated

The model is a 1D convolutional network over 40-day windows of three feature families (the same kinds of inputs the papers use):

- **On-chain metrics**: MVRV, SOPR, NUPL, hashrate, hodl waves, miner balances and more. 24 daily CSVs from [bitcoin-data.com](https://bitcoin-data.com). **These files are not distributed with this repo**: bitcoin-data.com's terms don't allow redistributing downloaded data. [assets/FEATURES.md](assets/FEATURES.md) lists every file, the exact date ranges used, and how to download them yourself (free but manual).
- **Macro markets**: gold, oil, S&P 500, VIX, treasury futures and more (fetched automatically from Yahoo Finance)
- **Technical indicators**: RSI, MACD, Bollinger Bands, SMAs/EMAs, ATR, OBV, VWAP (computed from the OHLCV data)

See [MODEL.md](MODEL.md) for the formula-by-formula specification.

```
assets/*.csv ─┐
Yahoo Finance ┼─► data.py ──► model.py ──► results/best_test_dataset.csv ──► backtest.py ──► results/
              ┘   (dataset)   (train)      (predictions)                     (evaluate)      (metrics + charts)
```

The evaluation is built to make lying hard:

- chronological 60/20/20 split; the validation region runs as 3 walk-forward folds, each training on everything before its slice
- causal rolling z-scores (90-day trailing window),  no scaling statistic ever sees the future
- LightGBM feature selection refit inside each fold, on that fold's training data only
- hyperparameter search (400 random trials) selects on *mean validation backtest Sharpe including fees*, not accuracy
- the test set is evaluated exactly once, by the single search winner

The result (`results/best_params.json`): validation Sharpe 3.27 collapses to **test Sharpe 1.29**, validation accuracy 56.1% to **test accuracy 52.2%** against an always-long base rate of 51.5%. That gap is selection bias made visible. The best of 400 trials always looks great on the data that chose it. And the winning model is barely a model at all: it opens the test window short, flips long on 2024-08-10, and stays long for the remaining 410 of 433 days. The confusion matrix shows the degeneracy directly; 410 of 433 predictions are "up":

![Confusion matrix on the test set](results/confusion_matrix.png)

## Backtest

Test window Jul 2024 → Sep 2025 (433 days), $1,000 start, 0.1% taker fee per side, equity marked to market daily (open positions valued at every close, so drawdowns during a trade are real, not just booked-at-exit):

| Strategy | Return | CAGR | Sharpe | Max drawdown | Win rate | Trades |
|----------|-------:|-----:|-------:|-------------:|---------:|-------:|
| **Model signal** | **+99%** | 79% | 1.44 | -28% | 100% | **2** |
| Buy & hold | +75% | 60% | 1.17 | -28% | 51% | — |
| Bollinger (long+short) | +71% | 60% | 1.69 | -12% | 65% | 23 |
| Bollinger (long only) | +53% | 45% | 1.60 | -7% | 80% | 10 |
| Bollinger (short only) | +12% | 11% | 0.46 | -12% | 54% | 13 |

This time the model signal *beats* buy-and-hold and that is the trap. The entire edge is one trade: the model opens the test window short, happens to catch the early-August 2024 crash, flips long on 2024-08-10 and never trades again. Two trades, 100% win rate, infinite profit factor; numbers that clean are a sample-size warning, not an edge. The direction calls behind it are 52.2% accurate against a 51.5% always-long base rate, so the outperformance is one lucky flip, not skill; a search seeded differently would crown a winner whose single flip lands somewhere less fortunate.

![Model signal equity curve](results/equity_model_signal.png)

![Model signal trades](results/trades_model_signal.png)

- **Model signal**: always in the market. Long while the model predicts up, short when it predicts down.
- **Bollinger Bands**: mean reversion. Buy below the lower band, short above the upper band, exit at the middle band or a 3% stop loss.
- **Buy & hold**: the baseline every crypto strategy must beat.

Metrics per strategy: total return, CAGR, Sharpe, Sortino, Calmar, max drawdown, profit factor and win rate (see `evaluate()` in `backtest.py`).

## Quick start

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python backtest.py        # seconds, offline: backtests the included predictions,
                          # writes results/metrics.csv + charts
python test_backtest.py   # sanity check, prints "ok"
python leakage_demo.py    # ~10 min on CPU: the honest-vs-bugs accuracy table
python model.py 400      # optional: retrain with 400 search trials
                          # (hours on CPU; defaults to 50 if omitted)
```

Trained artifacts and predictions are included, so `backtest.py` and `test_backtest.py` work immediately with no data downloads. `leakage_demo.py` and `model.py` rebuild the dataset, which requires the on-chain CSVs (follow [assets/FEATURES.md](assets/FEATURES.md) to download them first (they can't be redistributed here).

## Project structure

```
├── assets/            # input data location
│   └── FEATURES.md    # every metric to download, exact filenames + date ranges
├── results/           # everything the pipeline produces
├── MODEL.md           # the mathematics of the pipeline, formula by formula
├── data.py            # dataset assembly + feature engineering
├── model.py           # training + hyperparameter search
├── backtest.py        # strategies, metrics, charts
├── leakage_demo.py    # reproduces paper-level accuracy via evaluation bugs
├── test_backtest.py   # self-check for the strategy logic
├── requirements.txt
└── LICENSE            # MIT
```

## Honest limitations

Fees are modeled (0.1% per side) but slippage, funding costs on shorts, and margin mechanics are not. Everything rests on one test window in one market regime. On-chain metrics are consumed at each day's close, though in production some publish with a lag. None of this changes the conclusion, it only makes the negative result more negative.

This is a research project, not financial advice. The finding is the product: next-day BTC direction is not predictable from public daily data with this approach, and claims otherwise deserve a hard look at their target alignment.
