"""Two alternative numeric classifiers with bounded CPU and tree capacity."""

from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PARAMETERS = {
    "catboost": {
        "iterations": 600, "depth": 6, "learning_rate": 0.08,
        "l2_leaf_reg": 15, "loss_function": "MultiClass", "boosting_type": "Plain",
        "bootstrap_type": "Bayesian", "bagging_temperature": 0.5,
        "random_strength": 1.0, "border_count": 128, "random_seed": 42,
        "allow_writing_files": False, "verbose": False,
    },
    "extra_trees": {
        "n_estimators": 128, "max_depth": 20, "min_samples_leaf": 128,
        "max_features": "sqrt", "random_state": 42,
    },
}


def create_model(name: str, threads: int = 4, smoke: bool = False) -> Any:
    """The smoke model differs only in its tree count and minimum leaf size."""
    parameters = PARAMETERS[name].copy()
    if name == "catboost":
        if smoke:
            parameters["iterations"] = 3
        return CatBoostClassifier(**parameters, thread_count=threads)
    if smoke:
        parameters.update(n_estimators=3, min_samples_leaf=2)
    return Pipeline([
        ("missing", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True, keep_empty_features=True)),
        ("model", ExtraTreesClassifier(**parameters, n_jobs=threads)),
    ])


def fit_model(name: str, model: Any, features: np.ndarray, truth: np.ndarray) -> None:
    """Apply the same inverse-frequency square-root weights as the control."""
    counts = np.bincount(truth)
    if (counts == 0).any():
        raise ValueError("Training data must include every class")
    weights = np.sqrt(len(truth) / (len(counts) * counts))[truth]
    if name == "catboost":
        model.fit(features, truth, sample_weight=weights)
    else:
        model.fit(features, truth, model__sample_weight=weights)
