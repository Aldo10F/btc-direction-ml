# How the model works, mathematically

This documents every transformation between a raw CSV and a number in the results tables, in the order the pipeline applies them. File references point at the implementation so each formula can be checked against the code.

## 1. Notation

Let $C_t$ be the BTC close on day $t$, and $x_t \in \mathbb{R}^{F}$ the raw feature vector for day $t$: OHLCV, on-chain metrics (including the 13 HODL-wave age bands), technical indicators, and macro closes from Yahoo Finance. $F \approx 70$ columns, assembled in `data.py`.

## 2. Label

$$y_t = \mathbb{1}\!\left[\, C_{t+1} > C_t \,\right]$$

Day $t$ is labeled 1 if the *next* close is higher. The final row of the dataset is dropped because its label would need a close that does not exist yet (`load_dataset()` in `data.py`). Over the effective dataset the classes are close to balanced (~51% up), so the trivial "always predict up" classifier is the accuracy floor any real model must beat.

## 3. Causal normalization

Each feature column $j$ is turned into a rolling z-score using only a trailing 90-day window (`prepare_data()` in `model.py`):

$$z_{t,j} = \operatorname{clip}\!\left( \frac{x_{t,j} - \mu_{t,j}}{\sigma_{t,j} + 10^{-9}},\; -5,\; 5 \right), \qquad \mu_{t,j},\ \sigma_{t,j} \ \text{computed over } \{x_{s,j} : t-89 \le s \le t\}$$

Two properties matter:

- **Causality.** $z_{t,j}$ depends only on data up to day $t$. A scaler fit on the full dataset (the standard `StandardScaler` pattern) would leak the future's mean and variance into the past. That is bug A in `leakage_demo.py`.
- **Stationarity.** Level features (price, hashrate, supply) trend over years, so their raw values in the test period lie outside the training range. A trailing z-score expresses every feature as "how unusual is today relative to the last 90 days," which keeps test inputs in-distribution.

The first 90 rows are warm-up and are discarded.

## 4. Feature selection

Inside each walk-forward fold, a LightGBM classifier is fit on that fold's training rows only, and features are ranked by split importance; the top $k$ are kept ($k$ is a searched hyperparameter). Gradient-boosted trees measure how often each feature actually earns its place in a split, which handles the heavily correlated feature set (five SMAs/EMAs, two MVRV variants, …) better than univariate filters. Because the selector never sees validation or test rows, selection cannot launder future information into the model.

## 5. Sequences

Samples are sliding windows of $L$ consecutive normalized days (`create_sequences()` in `model.py`):

$$S_t = \left( z_{t-L+1}, \ldots, z_t \right) \in \mathbb{R}^{L \times k}, \qquad \text{label } y_t$$

The window ends on day $t$ and the label is day $t$'s move to day $t+1$: features are fresh through the moment the position would be taken, and the labeled move starts strictly after the window ends. Compare bug B in `leakage_demo.py`, where the label is shifted so the move it encodes already sits inside the window. A one-line indexing difference and the single largest accuracy inflator we found.

## 6. The network

A 1-D convolutional classifier (`build_model()` in `model.py`), reading $S_t$ as a length-$L$ signal with $k$ channels:

**Conv1D (×2).** Layer 1 with $m_1$ filters and kernel width $w$ (same padding), layer 2 with $m_2$ filters:

$$h^{(1)}_{t,f} = \operatorname{ReLU}\!\left( b_f + \sum_{\tau = -\lfloor w/2 \rfloor}^{\lfloor w/2 \rfloor} \sum_{j=1}^{k} W_{\tau,j,f}\, z_{t+\tau,\,j} \right)$$

and analogously $h^{(2)}$ over $h^{(1)}$. Stacking two width-$w$ kernels gives each output position a receptive field of $2w-1$ days: the network detects **local temporal patterns** (a few days of joint feature behavior), not window-position-specific rules.

**Global average pooling.** $g_f = \frac{1}{L} \sum_{t=1}^{L} h^{(2)}_{t,f}$.

This is the architectural decision with the most explanatory power for our results. GAP averages each filter's activation over the whole window, so the classifier head sees only *"how strongly did pattern $f$ fire on average"*,  never *where* it fired. A signal parked at a fixed position (e.g., a leak living in the window's last row, as in bug B) is diluted by a factor of $L$ and largely invisible. That is why the CNN scores ~53% under bug B while a tabular LightGBM, which reads the leaking column directly, scores ~97% on the identical misalignment.

**Head.** A dense ReLU layer of $d$ units with dropout rate $p$, then a sigmoid:

$$\hat{p}_t = \sigma\!\left( w^\top a_t + b \right) = \Pr(y_t = 1 \mid S_t), \qquad \hat{y}_t = \mathbb{1}[\hat{p}_t > 0.5]$$

**Training.** Binary cross-entropy $\mathcal{L} = -\frac{1}{N}\sum_t \left[ y_t \log \hat{p}_t + (1-y_t) \log (1-\hat{p}_t) \right]$, minimized with Adam; early stopping on validation loss restores the best-epoch weights.

All architecture and training knobs ($L$, $k$, $m_1$, $m_2$, $w$, $d$, $p$, learning rate, batch size, patience) are hyperparameters, see `PARAM_DIST` in `model.py`.

## 7. Walk-forward validation

The data is split chronologically 60/20/20 (train / validation / test). The validation region is cut into 3 consecutive folds; fold $i$ trains on everything before its slice (so later folds train on more data, as a live system would) and predicts only its own slice. Feature selection is refit per fold on that fold's training rows.

One boundary subtlety is acknowledged rather than engineered away: the last training day before each fold (and before the test set) is labeled with its move into the *first* day of the next slice, so that single label uses one close from beyond the training boundary. Features remain strictly causal everywhere. The overlap is one label per boundary, and none of the evaluated (validation or test) returns are involved. A one-day embargo between slices would remove even this; with daily data and slices hundreds of days long, its effect is far below the noise floor of Section 8.

The score a hyperparameter trial competes on is **not accuracy** but the mean, over folds, of the annualized Sharpe of actually trading the fold's predictions (`signal_sharpe()` in `model.py`). With position $\pi_i = +1$ if the model predicts up, $-1$ otherwise, per-day strategy returns are

$$r_i = \pi_i \cdot \frac{C_{i+1} - C_i}{C_i} \;-\; c_i, \qquad c_i = \begin{cases} 2\phi & \text{if } \pi_{i+1} \neq \pi_i \ \text{(close + reopen)} \\ 0 & \text{otherwise} \end{cases}$$

plus one fee $\phi = 0.1\%$ each on the first entry and final exit, and

$$\text{Sharpe} = \frac{\overline{r - r_f/365}}{\operatorname{sd}(r)} \cdot \sqrt{365}$$

Selecting on fee-inclusive Sharpe instead of accuracy kills a whole class of degenerate winners: a model that is right 53% of the time but flips position daily loses its edge to fees ($2\phi$ per flip), and a model whose few correct calls carry the big moves beats one that wins many tiny days and misses crashes.

## 8. Selection bias, quantified

Random search draws $N$ trials (400 in the published run) and keeps the best mean validation Sharpe. Even if every trial had zero true skill and its measured Sharpe were pure noise, $\text{Sharpe}_n \sim \mathcal{N}(0, \sigma^2)$, the expected winner is

$$\mathbb{E}\!\left[\max_{n \le N} \text{Sharpe}_n\right] \approx \sigma \sqrt{2 \ln N} \;\; \approx\; 3.5\,\sigma \ \text{for } N = 400$$

A validation Sharpe around 3 from a 400-trial search over ~500 noisy validation days is therefore **expected under the null of no skill**, it is what the maximum of four hundred coin-flip strategies looks like. This is exactly why the pipeline evaluates the test set **once**, by the single search winner, after all selection is finished: the published validation→test collapse (Sharpe 3.3 → 1.3, accuracy 56% → 52%) is selection bias made visible, and it is the headline result. Any pipeline that lets test performance influence any choice; features, architecture, early stopping, or which run gets published reintroduces this bias at full strength.

## 9. Backtest metrics

`backtest.py` marks equity to market daily (open positions are revalued at every close, so drawdowns during a trade are real). With daily equity $E_t$, daily returns $\rho_t = E_t/E_{t-1} - 1$, horizon $Y$ years, and risk-free rate $r_f = 4\%$:

$$\text{CAGR} = \left( \frac{E_{\text{end}}}{E_0} \right)^{1/Y} - 1 \qquad \text{Sharpe} = \frac{\overline{\rho - r_f/365}}{\operatorname{sd}(\rho)}\sqrt{365} \qquad \text{Sortino} = \frac{\overline{\rho - r_f/365}}{\operatorname{sd}(\rho \mid \rho < 0)}\sqrt{365}$$

$$\text{MaxDD} = \min_t \left( \frac{E_t}{\max_{s \le t} E_s} - 1 \right) \qquad \text{Calmar} = \frac{-\text{CAGR}}{\text{MaxDD}} \qquad \text{Profit factor} = \frac{\sum \text{wins}}{-\sum \text{losses}}$$

Note the Sortino denominator: $\operatorname{sd}(\rho \mid \rho < 0)$ is the standard deviation of the *negative-return days only*, which is exactly what `evaluate()` computes. The textbook Sortino instead uses the downside deviation $\sqrt{\frac{1}{N}\sum_t \min(\rho_t - r_f/365,\, 0)^2}$, averaged over **all** $N$ days and not centered. The two denominators differ, so the Sortino values in `metrics.csv` are only comparable with each other, not with figures computed under the textbook definition.

## 10. The null hypothesis

Under weak-form market efficiency, tomorrow's direction given today's public information is approximately a Bernoulli draw with $p \approx 0.5$ (plus a small drift that makes "always long" the best naive predictor). Any classifier's out-of-sample accuracy should then sit at $\max(p, 1-p)$, the base rate, regardless of architecture. That is the outcome this pipeline was built to detect honestly, and the outcome it found. A published accuracy of 70%+ on this target therefore demands an extraordinary explanation, and Section 5's one-line indexing bug supplies a perfectly ordinary one.
