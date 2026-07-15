"""
Reproduce the high accuracies that BTC-direction papers report, on this exact data and model, by
re-introducing common evaluation bugs:

  A. scaler fit on the FULL dataset + random shuffled split of overlapping sequences
     (the usually-blamed leaks. They barely move direction accuracy, because
     daily direction labels are near-independent coin flips)
  B. misaligned target: the label is a price move that already sits INSIDE the feature
     window, a one-line indexing bug, and the one that actually manufactures headline
     accuracy. Everything else (scaling, split) stays honest.

The honest arms use the same architecture and features with causal rolling z-scores and a
chronological 80/20 split. Any accuracy gap is the bug, not the model.

Run: python leakage_demo.py  →  results/leakage_demo.csv
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from data import RESULTS_DIR, load_dataset
from model import SEED, build_model, create_sequences

SEQ_LEN = 40
Z_WINDOW = 90


def cnn_accuracy(X_train, y_train, X_test, y_test):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    return accuracy_score(y_test, (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel())


if __name__ == "__main__":
    import lightgbm as lgb

    df = load_dataset()
    features = [c for c in df.columns if c != "Target"]
    results = {}

    # honest: causal rolling z-scores, chronological 80/20 split
    roll = df[features].rolling(Z_WINDOW)
    z = ((df[features] - roll.mean()) / (roll.std() + 1e-9)).clip(-5, 5)
    z = z.iloc[Z_WINDOW:].fillna(0.0).values
    y = df["Target"].values[Z_WINDOW:]
    split = int(len(z) * 0.8)
    X_train, y_train = create_sequences(z[:split], y[:split], SEQ_LEN)
    X_test, y_test = create_sequences(z[split:], y[split:], SEQ_LEN)
    results["honest CNN"] = cnn_accuracy(X_train, y_train, X_test, y_test)
    print(f"honest CNN: {results['honest CNN']:.4f}")

    # honest LightGBM baseline: same z-scores and split, day-t features only (no sequences)
    lgbm = lgb.LGBMClassifier(random_state=SEED, verbose=-1).fit(z[:split], y[:split])
    results["honest LightGBM"] = accuracy_score(y[split:], lgbm.predict(z[split:]))
    print(f"honest LightGBM: {results['honest LightGBM']:.4f}")

    # bug A: full-dataset scaler + shuffled random split
    z_leak = StandardScaler().fit_transform(df[features].fillna(0.0))
    X, ys = create_sequences(z_leak, df["Target"].values, SEQ_LEN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, ys, test_size=0.2, shuffle=True, random_state=SEED
    )
    results["bug A: full-data scaler + shuffled split"] = cnn_accuracy(
        X_train, y_train, X_test, y_test
    )
    print(f"bug A: {results['bug A: full-data scaler + shuffled split']:.4f}")

    # bug B: misaligned target, label y[i] is the close[i+L-1] vs close[i+L-2] move, which the
    # window rows i..i+L-1 already contain. Honest scaling and chronological split throughout.
    X_seq = np.array([z[i : i + SEQ_LEN] for i in range(len(z) - SEQ_LEN)])
    y_seen = y[SEQ_LEN - 2 : len(z) - 2]
    cut = split - SEQ_LEN
    results["bug B (CNN): label inside feature window"] = cnn_accuracy(
        X_seq[:cut], y_seen[:cut], X_seq[cut:], y_seen[cut:]
    )
    print(f"bug B CNN: {results['bug B (CNN): label inside feature window']:.4f}")

    # the same misalignment on the tabular baseline: label = the row's own daily move, which the
    # row's Daily_Return feature already encodes. The CNN above is largely immune because
    # GlobalAveragePooling1D is position-blind; LightGBM reads the leak directly.
    lgbm_mis = lgb.LGBMClassifier(random_state=SEED, verbose=-1).fit(z[1:split], y[: split - 1])
    results["bug B (LightGBM): label inside features"] = accuracy_score(
        y[split - 1 : -1], lgbm_mis.predict(z[split:])
    )
    print(f"bug B LightGBM: {results['bug B (LightGBM): label inside features']:.4f}")

    out = pd.Series(results, name="test_accuracy").round(4)
    out.to_csv(RESULTS_DIR / "leakage_demo.csv")
    print(f"\n{out}\nSaved leakage_demo.csv to {RESULTS_DIR}")
