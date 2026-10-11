"""Readable relationships between the current track and its prior profile."""

import numpy as np


def extra_features(matrix, columns, rows):
    """Build a small batch, keeping the existing memory-mapped cache intact."""
    index = {name: i for i, name in enumerate(columns)}

    def values(name):
        return np.asarray(matrix[rows, index[name]], dtype="float32")

    result = {}
    for name in ["total_distance_m", "spatial_spread_m", "num_points"]:
        current = np.log1p(values(name))
        usual = np.log1p(values(f"past_mean_{name}"))
        result[f"relative_log_{name}"] = current - usual
    for name in ["straightness", "turn_rate_45", "duplicate_ratio"]:
        result[f"deviation_{name}"] = values(name) - values(f"past_mean_{name}")
    for name in ["log_total_distance_m", "log_spatial_spread_m", "active_fraction"]:
        result[f"trend7_30_{name}"] = values(f"recent7_{name}") - values(
            f"recent30_{name}"
        )
    # Distances to prior quiet/active locations describe operating range.
    # A quiet location is not asserted to be a known port.
    lon, lat = values("center_lon"), values("center_lat")
    for state in ["quiet", "active"]:
        past_lon = values(f"behavior_past_mean_{state}_lon")
        past_lat = values(f"behavior_past_mean_{state}_lat")
        dlat = np.radians(lat - past_lat)
        dlon = np.radians(lon - past_lon)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(
            np.radians(past_lat)
        ) * np.sin(dlon / 2) ** 2
        distance = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        result[f"distance_from_prior_{state}_m"] = distance
    return result


def feature_names(columns, feature_set):
    if feature_set == "profile":
        return [
            name for name in columns
            if name.startswith(("past_", "behavior_past_", "recent"))
            or name == "days_since_previous_date"
        ]
    if feature_set == "behavior":
        return list(columns)
    if feature_set == "enriched":
        dummy = np.empty((0, len(columns)), dtype="float32")
        return list(columns) + list(extra_features(dummy, columns, np.array([], dtype=int)))
    raise ValueError(f"Unknown feature set: {feature_set}")


def feature_batch(matrix, columns, rows, feature_set):
    import pandas as pd

    names = feature_names(columns, feature_set)
    base = [name for name in names if name in columns]
    selected = [columns.index(name) for name in base]
    frame = pd.DataFrame(matrix[np.ix_(rows, selected)], columns=base)
    if feature_set == "enriched":
        for name, values in extra_features(matrix, columns, rows).items():
            frame[name] = values
    return frame
