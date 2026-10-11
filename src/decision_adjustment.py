"""Small, deterministic class-decision search; no probability calibration claim."""

import numpy as np

from src.improved_training import row_scores


def apply_class_weights(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scale class decision scores and normalize for storage."""
    weights = np.asarray(weights, dtype="float64")
    if weights.shape != (probabilities.shape[1],) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("One finite positive weight per class is required")
    adjusted = probabilities * weights
    return (adjusted / adjusted.sum(axis=1, keepdims=True)).astype("float32")


def fit_class_weights(truth: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Two coordinate passes, bounded weights; both fitting metrics must improve.

    Weights are decision parameters, not calibrated class probabilities.
    Call only with the designated calibration vessels, never evaluation labels.
    """
    weights = np.ones(probabilities.shape[1])
    floor = row_scores(truth, probabilities)
    best = (floor["accuracy"] + floor["macro_f1"]) / 2
    for _ in range(2):
        changed = False
        for index in range(len(weights)):
            selected = weights.copy()
            for step in [0.85, 1.15]:
                proposal = weights.copy()
                proposal[index] = np.clip(proposal[index] * step, 0.7, 1.4)
                scores = row_scores(truth, probabilities * proposal)
                joint = (scores["accuracy"] + scores["macro_f1"]) / 2
                if all(scores[key] >= floor[key] for key in floor) and joint > best + 1e-6:
                    best, selected = joint, proposal
            changed |= not np.array_equal(selected, weights)
            weights = selected
        if not changed:
            break
    return weights
