"""Direction classifier (Conv1D) with LightGBM feature selection and random search."""

import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from backtest import FEE
from data import PREDICTIONS_CSV, RESULTS_DIR

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_FILE = RESULTS_DIR / "best_cnn_model.keras"

PARAM_DIST = {
    "sequence_length": list(range(10, 90, 10)),
    "n_features": list(range(20, 80, 5)),
    "conv1_filters": [32, 64, 128],
    "conv2_filters": [16, 32, 64],
    "dense_units": [64, 128, 256],
    "dropout": [0.3, 0.5],
    "learning_rate": [0.001, 0.0005],
    "batch_size": [32, 64],
    "patience": [5, 10],
    "kernel_size": [3, 5],
}


def select_features_lgbm(X_train, y_train, feature_names, n_features):
    model = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.nlargest(n_features).index.tolist()


def create_sequences(X, y, sequence_length):
    """Each sample is labeled with its window's LAST row's target (the move from that row's close
    to the next close), so features are fresh through the day the position is taken."""
    xs = np.array([X[i : i + sequence_length] for i in range(len(X) - sequence_length + 1)])
    return xs, np.asarray(y[sequence_length - 1 :])


def signal_sharpe(close, preds, fee=FEE, risk_free_rate=0.04):
    """Annualized Sharpe of the always-in-market flip strategy (backtest.predicted_target_strategy):
    position taken at each close per the prediction, return realized at the next close. Each flip
    costs 2*fee of notional (close + reopen); entry and final close cost fee each."""
    pos = np.where(np.asarray(preds) == 1, 1.0, -1.0)
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return 0.0
    rets = pos[:-1] * (np.diff(close) / close[:-1])
    costs = np.where(pos[1:] != pos[:-1], 2 * fee, 0.0)
    costs[0] += fee
    costs[-1] += fee
    rets = rets - costs
    excess = rets - risk_free_rate / 365
    return float(excess.mean() / rets.std() * 365**0.5) if rets.std() > 0 else 0.0


def prepare_data(
    df,
    target_column="Target",
    sequence_length=10,
    n_features=50,
    val_size=0.2,
    test_size=0.2,
    n_folds=3,
    z_window=90,
):
    """Walk-forward splits. Features are causal rolling z-scores (each column normalized by its own
    trailing window), so level features stay in-distribution out of sample and no scaling statistic
    ever sees the future. The validation region is split into n_folds consecutive slices; each fold
    trains on everything before its slice, with feature selection refit on that train data only.
    The last fold's model and feature set are the final ones, so test sequences use the last fold's
    selection."""
    df = df.sort_index()
    features = [c for c in df.columns if c != target_column]
    roll = df[features].rolling(z_window)
    z = ((df[features] - roll.mean()) / (roll.std() + 1e-9)).clip(-5, 5)
    zv = z.iloc[z_window:].fillna(0.0).values
    raw = df.iloc[z_window:]

    i_val = int(len(raw) * (1 - val_size - test_size))
    i_test = int(len(raw) * (1 - test_size))

    le = LabelEncoder()
    le.fit(raw[target_column].iloc[:i_val])
    assert len(le.classes_) == 2, "Target must be binary"
    y = le.transform(raw[target_column])
    close = raw["close"].values

    folds = []
    bounds = np.linspace(i_val, i_test, n_folds + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        selected = select_features_lgbm(zv[:a], y[:a], features, min(n_features, len(features)))
        cols = [features.index(f) for f in selected]
        folds.append(
            {
                "train": create_sequences(zv[:a, cols], y[:a], sequence_length),
                "val": create_sequences(zv[a:b, cols], y[a:b], sequence_length),
                "val_close": close[a + sequence_length - 1 : b],
                "selected": selected,
                "cols": cols,
            }
        )

    cols = folds[-1]["cols"]
    return {
        "folds": folds,
        "le": le,
        "selected": folds[-1]["selected"],
        "test": create_sequences(zv[i_test:, cols], y[i_test:], sequence_length),
        "test_close": close[i_test + sequence_length - 1 :],
        "test_indices": raw.index[i_test + sequence_length - 1 :],
        "test_df": raw.iloc[i_test:],
    }


def build_model(
    input_shape,
    conv1_filters=64,
    conv2_filters=32,
    dense_units=128,
    dropout=0.5,
    learning_rate=0.001,
    kernel_size=3,
):
    model = Sequential(
        [
            Conv1D(
                conv1_filters,
                kernel_size,
                activation="relu",
                padding="same",
                input_shape=input_shape,
            ),
            Conv1D(conv2_filters, kernel_size, activation="relu", padding="same"),
            GlobalAveragePooling1D(),
            Dense(dense_units, activation="relu"),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def random_search(df, target_column="Target", num_trials=50, epochs=50, n_folds=3):
    """Trials compete on the MEAN validation backtest Sharpe (flip strategy, fees included) across
    walk-forward folds; the kept model is the last fold's (trained on the most data). The test set
    is evaluated exactly once, by the winning model, at the end."""
    best = {"val_sharpe": float("-inf"), "model": None, "params": None, "data": None}
    for trial in range(num_trials):
        params = {k: random.choice(v) for k, v in PARAM_DIST.items()}
        print(f"\nTrial {trial + 1}/{num_trials} | {params}")
        if params["sequence_length"] >= len(df) * 0.2 / n_folds - 10:
            print("Skipping: sequence too long for a walk-forward fold")
            continue

        data = prepare_data(
            df, target_column, params["sequence_length"], params["n_features"], n_folds=n_folds
        )
        sharpes, accuracies, model = [], [], None
        for fold in data["folds"]:
            X_train, y_train = fold["train"]
            X_val, y_val = fold["val"]

            model = build_model(
                (X_train.shape[1], X_train.shape[2]),
                params["conv1_filters"],
                params["conv2_filters"],
                params["dense_units"],
                params["dropout"],
                params["learning_rate"],
                params["kernel_size"],
            )
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=params["batch_size"],
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=params["patience"],
                        restore_best_weights=True,
                    )
                ],
            )

            val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).ravel()
            accuracies.append(accuracy_score(y_val, val_pred))
            sharpes.append(signal_sharpe(fold["val_close"], data["le"].inverse_transform(val_pred)))

        val_sharpe = float(np.mean(sharpes))
        val_accuracy = float(np.mean(accuracies))
        folds_str = ", ".join(f"{s:.2f}" for s in sharpes)
        print(
            f"Validation sharpe: {val_sharpe:.4f} "
            f"(folds: [{folds_str}], accuracy: {val_accuracy:.4f})"
        )
        if val_sharpe > best["val_sharpe"]:
            best.update(
                val_sharpe=val_sharpe,
                val_accuracy=val_accuracy,
                model=model,
                params=params,
                data=data,
            )
            print("New best (by mean validation sharpe)")

    if best["model"] is None:
        return best

    # the only test-set evaluation in the whole run
    data = best["data"]
    X_test, y_test = data["test"]
    y_pred = (best["model"].predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    best["accuracy"] = accuracy_score(y_test, y_pred)
    best["test_sharpe"] = signal_sharpe(data["test_close"], data["le"].inverse_transform(y_pred))
    best["cm"] = confusion_matrix(y_test, y_pred)
    best["features"] = data["selected"]

    # backtest.py runs offline from this CSV, so always include OHLCV + bands
    keep = ["open", "high", "low", "close", "volume", "bb_high", "bb_low", "bb_mid"]
    cols = list(dict.fromkeys(keep + data["selected"]))
    out = data["test_df"].loc[data["test_indices"], cols + [target_column]].copy()
    out["Predicted_" + target_column] = data["le"].inverse_transform(y_pred)
    out.sort_index().to_csv(PREDICTIONS_CSV)
    best["model"].save(MODEL_FILE)

    print(f"\nBest validation sharpe: {best['val_sharpe']:.4f}")
    print(f"Test sharpe (evaluated once): {best['test_sharpe']:.4f}")
    print(f"Test accuracy: {best['accuracy']:.4f}")
    print(f"Params: {best['params']}")
    print(f"Features: {best['features']}")
    print(f"Confusion matrix (test):\n{best['cm']}")
    _save_run_summary(best)
    return best


def _save_run_summary(best):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(best["cm"], cmap="Blues")
    for (r, c), v in np.ndenumerate(best["cm"]):
        ax.text(c, r, str(v), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — test accuracy {best['accuracy']:.4f}")
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (RESULTS_DIR / "best_params.json").write_text(
        json.dumps(
            {
                "val_sharpe": best["val_sharpe"],
                "test_sharpe": best["test_sharpe"],
                "val_accuracy": best["val_accuracy"],
                "test_accuracy": best["accuracy"],
                **best["params"],
                "features": best["features"],
            },
            indent=2,
        )
    )
    print(f"Saved confusion_matrix.png and best_params.json to {RESULTS_DIR}")


if __name__ == "__main__":
    import sys

    from data import load_dataset

    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    random_search(load_dataset(), num_trials=trials)
