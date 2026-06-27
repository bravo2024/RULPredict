
"""model.py - lag-feature forecaster for remaining-useful-life prediction.
RUL data is cycle-indexed (not timestamped), so calendar-based features
(e.g. day-of-week, day-of-year) are spurious and have been removed.
Reference: Saxena et al. (2008) 'Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation' — the CMAPSS dataset."""
from __future__ import annotations
import numpy as np
from src.core import RidgeRegression, Standardizer, rmse, smape, mae, temporal_split

PREDICT_KIND = "timeseries"
LAGS = [1, 2, 3, 7, 14, 28]


def _feat(s):
    """Build feature matrix from lag values only (no spurious calendar features).

    RUL prediction uses cycle-indexed sensor readings; calendar features
    (day-of-week, day-of-year) have no causal relationship with
    remaining useful life and would introduce noise.  Reference:
    CMAPSS data uses a cycle counter, not timestamps."""
    s = np.asarray(s, float)
    if not np.isfinite(s).all():
        raise ValueError("Series contains NaN or inf values.")
    if len(s) <= max(LAGS):
        raise ValueError(f"Need more than {max(LAGS)} observations, got {len(s)}.")
    rows, tgt = [], []
    st = max(LAGS)
    for i in range(st, len(s)):
        rows.append([s[i - l] for l in LAGS])
        tgt.append(s[i])
    return np.array(rows), np.array(tgt)


def _holdout_split(X, y, test_size=0.2, gap=0):
    """Chronological split with an optional gap between train and test.
    The gap prevents the first test observation from leaking the last
    training target through its lag features."""
    sp = int(len(X) * (1 - test_size))
    train_end = max(sp - gap, 0)
    return X[:train_end], X[sp:], y[:train_end], y[sp:]


def fit_and_evaluate(data):
    """Train ridge regression with lag features on RUL data.

    Uses a chronological train/test split with a gap of max(LAGS)
    observations to prevent data leakage from the last training
    observation into the first test feature vector."""
    s = np.asarray(data["series"], float)
    if not np.isfinite(s).all():
        raise ValueError("Series contains NaN or inf values.")

    X, y = _feat(s)

    gap = max(LAGS)
    X_train, X_test, y_train, y_test = _holdout_split(X, y, test_size=0.2, gap=gap)

    scaler = Standardizer().fit(X_train)
    Xs_tr = scaler.transform(X_train)
    Xs_te = scaler.transform(X_test)

    m = RidgeRegression(alpha=1.0).fit(Xs_tr, y_train)
    pred = m.predict(Xs_te)

    metrics = {
        "n_train": int(len(Xs_tr)),
        "n_test": int(len(Xs_te)),
        "rmse": rmse(y_test, pred),
        "smape_pct": smape(y_test, pred),
        "mae": mae(y_test, pred),
    }
    model_dict = {
        "model": m,
        "scaler": scaler,
        "lags": LAGS,
        "tail": s[-max(LAGS):].tolist(),
    }
    return model_dict, metrics


def forecast_next(model_dict, series=None):
    """Generate a one-step-ahead RUL forecast."""
    s = np.asarray(series if series is not None else model_dict.get("tail", []), float)
    if len(s) < max(model_dict["lags"]):
        raise ValueError(
            f"Need at least {max(model_dict['lags'])} observations for forecast, "
            f"got {len(s)}."
        )
    f = [s[-l] for l in model_dict["lags"]]
    f_scaled = model_dict["scaler"].transform(np.array([f]))
    return float(model_dict["model"].predict(f_scaled)[0])
