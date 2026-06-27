"""data.py - real and synthetic health-index series for RULPredict."""
from __future__ import annotations
import numpy as np
from pathlib import Path


def fetch_ai4i():
    """Load AI4I 2020 Predictive Maintenance dataset from UCI ML Repository.

    10,000 machine operating cycles from a milling machine.  Tool wear [min]
    accumulates from 0 to 253 min (forced replacement threshold).  Converted
    to a health index: HI = 100 * (1 - wear/253), so the series starts near
    100 (new tool) and degrades toward 0 (worn out / failure).

    Reference: S. Matzka (2020) 'Explainable Artificial Intelligence for
    Predictive Maintenance Applications', IEEE ICMLA 2020.
    Source: UCI ML Repository, dataset #601.
    """
    import pandas as pd
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00601/ai4i2020.csv")
    df = pd.read_csv(url, encoding="utf-8-sig")
    wear = df["Tool wear [min]"].astype(float).to_numpy()
    max_wear = 253.0
    health = np.clip(100.0 * (1.0 - wear / max_wear), 0.0, 100.0)
    return {
        "series": health,
        "source": "AI4I 2020 Predictive Maintenance — UCI ML Repository",
        "n_cycles": len(health),
    }


def make_synthetic(n=240, seed=42):
    """Synthetic turbofan health-index with degradation + noise (fallback)."""
    rng = np.random.default_rng(seed)
    cycles = np.arange(n)
    degradation = 100 - 0.16 * cycles - 0.00055 * cycles ** 2
    operating_cycle = (2.5 * np.sin(2 * np.pi * cycles / 30)
                       + 1.2 * np.sin(2 * np.pi * cycles / 7))
    noise = rng.normal(0, 1.4, n)
    health_index = np.clip(degradation + operating_cycle + noise, 0, 100)
    return {"series": health_index, "cycles": cycles, "source": "Synthetic"}


def load_real(csv_name, value_col, date_col=None):
    csv_path = Path("data/raw") / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    import pandas as pd
    df = pd.read_csv(csv_path)
    if date_col:
        df = df.sort_values(date_col)
    arr = df[value_col].astype(float).to_numpy()
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    return {"series": arr}
