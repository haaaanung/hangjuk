"""Combine a vessel's predictions using strictly earlier calendar dates.

Inputs contain predictions and observation keys only, never class labels.
Each observed day receives one vote so busy days do not dominate the history.
"""

import numpy as np
import pandas as pd


def prior_prediction_history(keys, probabilities):
    """Return the prior-day mean probability and number of observed prior days.

    Supply all available rows for each vessel, including its earlier dates.
    Row order is preserved. Other rows on the current date are excluded.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    if probabilities.ndim != 2 or len(keys) != len(probabilities):
        raise ValueError("Observation keys and probability rows must align")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    key_columns = ["rfid_id", "part_dt"]
    if keys[key_columns].isna().any().any():
        raise ValueError("Missing vessel/date keys")
    columns = [f"p_{i}" for i in range(probabilities.shape[1])]
    table = keys[key_columns].reset_index(drop=True).copy()
    table[columns] = probabilities
    daily = table.groupby(key_columns, sort=True)[columns].mean()
    groups = daily.groupby(level="rfid_id", sort=False)
    days = groups.cumcount().to_numpy()
    # Subtract the entire current day, including all its other observations.
    prior_sum = groups.cumsum().to_numpy() - daily.to_numpy()
    prior_mean = np.divide(
        prior_sum,
        days[:, None],
        out=np.zeros_like(prior_sum),
        where=days[:, None] > 0,
    )
    positions = daily.index.get_indexer(pd.MultiIndex.from_frame(table[key_columns]))
    return prior_mean[positions].astype("float32"), days[positions]


def blend_history(probabilities, prior_mean, prior_days, strength, warmup_days=3):
    """History influence grows gradually; a vessel's first day is unchanged."""
    if not 0 <= strength <= 1 or warmup_days <= 0:
        raise ValueError("strength must be in [0, 1]; warmup_days must be positive")
    days = np.asarray(prior_days)
    weight = (strength * days / (days + warmup_days))[:, None]
    result = (1 - weight) * probabilities + weight * prior_mean
    result = np.maximum(result, 0)
    return (result / result.sum(axis=1, keepdims=True)).astype("float32")
