"""
Robustness checks on the published test window.
Run: python robustness.py

The headline conclusion rests on one 433-day test window. Retraining on other regimes is not
possible without re-running the search (and the on-chain data only reaches back so far), but the
fragility of the model signal's "win" inside that window can be quantified directly:

1. Sub-windows: model signal vs buy-and-hold over consecutive 90-day slices of the test window,
   each traded standalone. Shows where the outperformance actually lives.
2. Flip-day sweep: the winning model is a single short->long flip, so enumerate every possible
   flip day and backtest each path. Shows where the published flip ranks among all its siblings.
3. Binomial test: the probability of scoring >= the observed accuracy by predicting at the
   always-long base rate with zero skill.
4. Random baselines: 1000 seeded random strategies each, backtested with and without fees.
   Coin-flip (P(long)=0.5) shows what PnL pure chance earns in this window's trend; the
   bias-matched variant (P(long) = the model's own long fraction) rides the trend like the
   model does, isolating whether its timing adds anything beyond its long bias.

Outputs: results/robustness.csv, results/robustness.json, results/robustness_flip_days.png,
results/robustness_random.png
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from backtest import hodl_strategy, predicted_target_strategy
from data import PREDICTIONS_CSV, RESULTS_DIR

SUB_WINDOW_DAYS = 90
BALANCE = 1000.0
N_RANDOM = 1000
SEED = 42


def _final_balance(trades):
    return BALANCE if trades.empty else float(trades["balance"].iloc[-1])


def sub_window_table(df):
    """Model signal vs buy-and-hold, each slice backtested standalone."""
    rows = {}
    for start in range(0, len(df), SUB_WINDOW_DAYS):
        part = df.iloc[start : start + SUB_WINDOW_DAYS]
        label = f"{part.index[0]} → {part.index[-1]}"
        rows[label] = {
            "days": len(part),
            "model_return_%": (_final_balance(predicted_target_strategy(part)) / BALANCE - 1) * 100,
            "hodl_return_%": (_final_balance(hodl_strategy(part)) / BALANCE - 1) * 100,
            "accuracy_%": (part["Target"] == part["Predicted_Target"]).mean() * 100,
            "base_rate_%": part["Target"].mean() * 100,
        }
    return pd.DataFrame(rows).T


def flip_day_sweep(df):
    """Final balance of every single-flip path: short until day d, long from day d on.
    d=0 is always-long, d=len(df) is always-short. The published model is one of these paths."""
    paths = (df.assign(Predicted_Target=np.arange(len(df)) >= d) for d in range(len(df) + 1))
    return pd.Series([_final_balance(predicted_target_strategy(p)) for p in paths])


def random_baseline(df, p=0.5, n=N_RANDOM, seed=SEED):
    """Final balances of n random strategies: each day is long with probability p, else short.
    The same n signal paths are backtested twice, with fees and with fee=0."""
    rng = np.random.default_rng(seed)
    finals = {"fees": [], "no_fees": []}
    for _ in range(n):
        path = df.assign(Predicted_Target=rng.random(len(df)) < p)
        finals["fees"].append(_final_balance(predicted_target_strategy(path)))
        finals["no_fees"].append(_final_balance(predicted_target_strategy(path, fee=0)))
    return pd.DataFrame(finals)


def main():
    df = pd.read_csv(PREDICTIONS_CSV, index_col=0)
    df.index = pd.to_datetime(df.index).date

    table = sub_window_table(df)
    table.round(2).to_csv(RESULTS_DIR / "robustness.csv")
    print("=== model vs buy-and-hold per 90-day sub-window ===")
    print(table.round(2))

    finals = flip_day_sweep(df)
    # the published model IS one of these paths: short until its one flip, long after
    published_day = int(np.argmax(df["Predicted_Target"].values == 1))
    published_final = finals[published_day]
    always_long = finals[0]
    summary = {
        "paths": len(finals),
        "published_flip_day": int(published_day),
        "published_flip_date": str(df.index[published_day]),
        "published_final_balance": round(float(published_final), 2),
        "always_long_final_balance": round(float(always_long), 2),
        "paths_beating_always_long": int((finals > always_long).sum()),
        "published_percentile": round(float((finals <= published_final).mean() * 100), 1),
        "median_final_balance": round(float(finals.median()), 2),
    }

    correct = int((df["Target"] == df["Predicted_Target"]).sum())
    base_rate = float(df["Target"].mean())
    test = binomtest(correct, len(df), p=base_rate, alternative="greater")
    summary["binomial"] = {
        "correct": correct,
        "days": len(df),
        "accuracy_%": round(correct / len(df) * 100, 1),
        "base_rate_%": round(base_rate * 100, 1),
        "p_value": round(float(test.pvalue), 3),
    }

    model_final = {
        "fees": published_final,
        "no_fees": _final_balance(predicted_target_strategy(df, fee=0)),
    }
    hodl_final = _final_balance(hodl_strategy(df))
    long_frac = float((df["Predicted_Target"] == 1).mean())
    baselines = {
        "coin_flip": random_baseline(df),
        "bias_matched": random_baseline(df, p=long_frac),  # same long bias as the model
    }
    summary["random_baseline"] = {
        "sims": N_RANDOM,
        "seed": SEED,
        "hodl_final_balance": round(hodl_final, 2),
    }
    for name, rand in baselines.items():
        stats = {"p_long": round(0.5 if name == "coin_flip" else long_frac, 3)}
        for kind in ("fees", "no_fees"):
            r = rand[kind]
            stats[kind] = {
                "model_final_balance": round(float(model_final[kind]), 2),
                "median_final_balance": round(float(r.median()), 2),
                "mean_final_balance": round(float(r.mean()), 2),
                "pct_positive_pnl": round(float((r > BALANCE).mean() * 100), 1),
                "pct_beating_model": round(float((r > model_final[kind]).mean() * 100), 1),
                "model_percentile": round(float((r <= model_final[kind]).mean() * 100), 1),
            }
        summary["random_baseline"][name] = stats

    (RESULTS_DIR / "robustness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== flip-day sweep + binomial test + random baseline ===")
    print(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = list(df.index) + [df.index[-1]]  # path d flips on day d; d=len(df) never goes long
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, finals.values, color="#1f77b4", lw=2)
    ax.axhline(always_long, color="#888888", ls="--", lw=1)
    ax.annotate(
        f"always long: ${always_long:,.0f}",
        (dates[len(dates) // 2], always_long),
        textcoords="offset points",
        xytext=(0, 6),
        color="#555555",
    )
    ax.plot([dates[published_day]], [published_final], "o", ms=9, color="#ff7f0e")
    ax.annotate(
        f"published model: ${published_final:,.0f}\n({summary['published_percentile']:.0f}th "
        f"percentile of all flips)",
        (dates[published_day], published_final),
        xytext=(dates[len(dates) // 3], 2150),
        color="#b45309",
        arrowprops={"arrowstyle": "->", "color": "#b45309"},
    )
    ax.set_title("Final balance of every single short→long flip path in the test window")
    ax.set_ylabel("Final balance (USD)")
    ax.set_xlabel("Flip day (short before, long after)")
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS_DIR / "robustness_flip_days.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    coin, bias = baselines["coin_flip"], baselines["bias_matched"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.histogram_bin_edges(pd.concat([coin["fees"], coin["no_fees"], bias["fees"]]), bins=50)
    ax.hist(coin["no_fees"], bins=bins, alpha=0.6, color="#1f77b4", label="coin-flip, no fees")
    ax.hist(coin["fees"], bins=bins, alpha=0.6, color="#888888", label="coin-flip, with fees")
    ax.hist(
        bias["fees"],
        bins=bins,
        alpha=0.6,
        color="#2ca02c",
        label=f"bias-matched (p_long={long_frac:.2f}), with fees",
    )
    ax.axvline(BALANCE, color="black", ls=":", lw=1, label="starting balance")
    ax.axvline(hodl_final, color="#555555", ls="--", lw=2, label=f"buy & hold (${hodl_final:,.0f})")
    ax.axvline(
        model_final["fees"],
        color="#ff7f0e",
        lw=2,
        label=f"model, with fees (${model_final['fees']:,.0f})",
    )
    ax.set_title(f"{N_RANDOM} random strategies per baseline vs the model (seed={SEED})")
    ax.set_xlabel("Final balance (USD)")
    ax.set_ylabel("Strategies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS_DIR / "robustness_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"\nSaved robustness.csv, robustness.json, robustness_flip_days.png, "
        f"robustness_random.png to {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
