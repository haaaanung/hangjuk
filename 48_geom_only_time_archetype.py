"""Forward-time geom-only route-archetype density ensemble."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "experiments" / "geom_only_time_archetype"
CACHE = ROOT / "outputs" / "cache" / "geom_only_time_archetype"
V1_CACHE = ROOT / "outputs" / "cache" / "geom_only_v1"
SPATIAL_CACHE = ROOT / "outputs" / "cache" / "geom_only_time_spatial"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def geom_column(code, matrix, base_indices, supplemental, name):
    index = code.GEOM_ONLY_COLUMNS.index(name)
    if index < len(base_indices):
        return np.asarray(matrix[:, base_indices[index]], dtype="float64")
    return np.asarray(supplemental[:, index - len(base_indices)], dtype="float64")


def bounded_bin(values, transform, maximum=63):
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.floor(transform(clean)), 0, maximum).astype("uint64")


def pack_signature(lon, lat, resolution, bins):
    x = np.clip(np.floor((np.nan_to_num(lon, nan=-180.0) + 180) / resolution), 0, 65535).astype("uint64")
    y = np.clip(np.floor((np.nan_to_num(lat, nan=-90.0) + 90) / resolution), 0, 32767).astype("uint64")
    key = x | (y << np.uint64(16))
    shift = 31
    for values in bins:
        key |= values << np.uint64(shift)
        shift += 6
    if shift > 61:
        raise ValueError("Archetype signature exceeds safe packed width")
    return key


def posterior(train_key, target_key, train_truth, smoothing):
    class_count = np.bincount(train_truth, minlength=4).astype("float64")
    prior = class_count / class_count.sum()
    pair = (train_key << np.uint64(2)) | train_truth.astype("uint64")
    unique_pair, counts = np.unique(pair, return_counts=True)
    unique_key = np.unique(train_key)
    table = np.zeros((len(unique_key), 4), dtype="float64")
    pair_key = unique_pair >> np.uint64(2)
    position = np.searchsorted(unique_key, pair_key)
    table[position, (unique_pair & np.uint64(3)).astype("int64")] = counts
    target_position = np.searchsorted(unique_key, target_key)
    matched = target_position < len(unique_key)
    matched[matched] &= unique_key[target_position[matched]] == target_key[matched]
    result = np.empty((len(target_key), 4), dtype="float64")
    result[:] = prior
    observed = table[target_position[matched]]
    result[matched] = (observed + smoothing * prior) / (
        observed.sum(axis=1, keepdims=True) + smoothing
    )
    return result, float(matched.mean()), len(unique_key)


def archetype_probability(train_rows, target_rows, truth, arrays):
    lon, lat = arrays["center_lon"], arrays["center_lat"]
    distance = bounded_bin(arrays["total_distance_m"], lambda x: np.log1p(np.maximum(x, 0)) / .35)
    straight = bounded_bin(arrays["straightness"], lambda x: x * 12)
    turn_rate = bounded_bin(arrays["turn_rate_45"], lambda x: x * 16)
    spread = bounded_bin(arrays["spatial_spread_m"], lambda x: np.log1p(np.maximum(x, 0)) / .35)
    points = bounded_bin(arrays["num_points"], lambda x: np.log1p(np.maximum(x, 0)) / .25)
    elongation = bounded_bin(arrays["shape_elongation"], lambda x: x * 12)
    mean_turn = bounded_bin(arrays["mean_direction_change_deg"], lambda x: x / 6)
    heading = bounded_bin(arrays["heading_axial_concentration"], lambda x: x * 16)
    moving = bounded_bin(arrays["moving_fraction_10m"], lambda x: x * 16)
    definitions = (
        (.025, (distance, straight, turn_rate), 30.0, "ground_motion"),
        (.05, (spread, points, elongation), 45.0, "ground_footprint"),
        (.10, (mean_turn, heading, moving), 70.0, "ground_steering"),
        (.25, (distance, mean_turn, elongation), 100.0, "region_shape"),
    )
    result = np.zeros((len(target_rows), 4), dtype="float64")
    diagnostics = []
    for resolution, bins, smoothing, name in definitions:
        train_key = pack_signature(lon[train_rows], lat[train_rows], resolution, [v[train_rows] for v in bins])
        target_key = pack_signature(lon[target_rows], lat[target_rows], resolution, [v[target_rows] for v in bins])
        probability, coverage, cells = posterior(train_key, target_key, truth[train_rows], smoothing)
        result += probability
        diagnostics.append({
            "expert": name, "resolution_degrees": resolution,
            "smoothing_rows": smoothing, "train_archetypes": cells,
            "target_coverage": coverage,
        })
    return (result / len(definitions)).astype("float32"), diagnostics


def normalize(values):
    return values / values.sum(axis=1, keepdims=True)


def macro_f1_fast(y, prediction):
    matrix = np.bincount(y * 4 + prediction, minlength=16).reshape(4, 4)
    tp = np.diag(matrix).astype("float64")
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * tp, denominator, out=np.zeros(4), where=denominator > 0).mean())


def ensemble(base, archetype, spatial, archetype_weight, spatial_weight, bias):
    values = (
        (1 - archetype_weight - spatial_weight) * base
        + archetype_weight * archetype + spatial_weight * spatial
    )
    return normalize(values * bias)


def score(y, probability):
    prediction = probability.argmax(axis=1)
    return macro_f1_fast(y, prediction), float(np.mean(y == prediction))


def tune(y, base, archetype, spatial):
    rng = np.random.default_rng(20270122)
    rows = rng.choice(len(y), size=min(200_000, len(y)), replace=False)
    y, base, archetype, spatial = y[rows], base[rows], archetype[rows], spatial[rows]
    grids = (
        np.arange(1.0, 1.81, .1), np.arange(1.0, 1.81, .1),
        np.arange(.5, 1.21, .1), np.arange(.7, 1.61, .1),
    )
    records = []
    for start_a, start_s in ((.2, .15), (.4, .1), (.1, .3), (.5, .25)):
        aw, sw = start_a, start_s
        bias = np.array([1.6, 1.5, .7, 1.0])
        for _ in range(4):
            best = None
            for value in np.arange(0, .71, .05):
                if value + sw > .9:
                    continue
                key = score(y, ensemble(base, archetype, spatial, value, sw, bias))
                if best is None or key > best[0]:
                    best = key, value
            aw = best[1]
            best = None
            for value in np.arange(0, .51, .05):
                if value + aw > .9:
                    continue
                key = score(y, ensemble(base, archetype, spatial, aw, value, bias))
                if best is None or key > best[0]:
                    best = key, value
            sw = best[1]
            for class_index, candidates in enumerate(grids):
                best = None
                for value in candidates:
                    trial = bias.copy()
                    trial[class_index] = value
                    key = score(y, ensemble(base, archetype, spatial, aw, sw, trial))
                    if best is None or key > best[0]:
                        best = key, value
                bias[class_index] = best[1]
        probability = ensemble(base, archetype, spatial, aw, sw, bias)
        macro, accuracy = score(y, probability)
        records.append({
            "archetype_weight": aw, "spatial_weight": sw,
            "coastal": bias[0], "gillnet": bias[1],
            "longline": bias[2], "trap": bias[3],
            "macro_f1": macro, "accuracy": accuracy,
        })
    table = pd.DataFrame(records).sort_values(["macro_f1", "accuracy"], ascending=False)
    return table.iloc[0], table


def apply_selected(base, archetype, spatial, selected):
    bias = np.array([selected.coastal, selected.gillnet, selected.longline, selected.trap])
    return ensemble(
        base, archetype, spatial, float(selected.archetype_weight),
        float(selected.spatial_weight), bias,
    )


def metric(name, y, probability):
    prediction = probability.argmax(axis=1)
    return {
        "variant": name, "rows": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    code = load_module("32_geom_only_model.py", "geom_only_v1")
    frame, matrix, base_indices, supplemental, labels, truth, _, snapshot = code.load_data()
    names = (
        "center_lon", "center_lat", "total_distance_m", "straightness", "turn_rate_45",
        "spatial_spread_m", "num_points", "shape_elongation",
        "mean_direction_change_deg", "heading_axial_concentration", "moving_fraction_10m",
    )
    arrays = {name: geom_column(code, matrix, base_indices, supplemental, name) for name in names}
    dates = frame.part_dt.to_numpy()
    jan_apr = np.flatnonzero(dates < 20260501)
    may = np.flatnonzero((dates >= 20260501) & (dates < 20260601))
    jan_may = np.flatnonzero(dates < 20260601)
    june = np.flatnonzero(dates >= 20260601)
    may_archetype, may_diagnostics = archetype_probability(jan_apr, may, truth, arrays)
    june_archetype, june_diagnostics = archetype_probability(jan_may, june, truth, arrays)
    np.save(CACHE / "may_probability.npy", may_archetype)
    np.save(CACHE / "june_probability.npy", june_archetype)

    may_base = np.load(SPATIAL_CACHE / "may_base_probability.npy", mmap_mode="r")
    may_spatial = np.load(SPATIAL_CACHE / "may_spatial_probability.npy", mmap_mode="r")
    june_spatial = np.load(SPATIAL_CACHE / "june_spatial_probability.npy", mmap_mode="r")
    june_base = np.mean([
        np.load(V1_CACHE / name, mmap_mode="r") for name in (
            "time_june_probability.npy", "time_june_probability_member_2.npy",
            "time_june_probability_member_3.npy",
        )
    ], axis=0, dtype="float64")
    selected, search = tune(truth[may], may_base, may_archetype, may_spatial)
    search.to_csv(OUT / "may_search.csv", index=False, encoding="utf-8-sig")
    may_final = apply_selected(may_base, may_archetype, may_spatial, selected)
    june_final = apply_selected(june_base, june_archetype, june_spatial, selected)
    np.save(CACHE / "june_ensemble_probability.npy", june_final.astype("float32"))
    old_bias = np.array([1.6, 1.5, .7, 1.0])
    metrics = pd.DataFrame([
        metric("v1_existing", truth[june], normalize(june_base * old_bias)),
        metric("spatial_previous", truth[june], normalize((.75 * june_base + .25 * june_spatial) * np.array([1.3, 1.3, .9, 1.1]))),
        metric("archetype_only", truth[june], june_archetype),
        metric("may_selected_archetype_ensemble", truth[june], june_final),
    ])
    classes = pd.DataFrame(classification_report(
        truth[june], june_final.argmax(axis=1), labels=np.arange(4), target_names=labels,
        output_dict=True, zero_division=0,
    )).T.loc[labels].reset_index(names="class")
    may_result = metric("may_selected_archetype_ensemble", truth[may], may_final)
    selected_record = {key: float(selected[key]) for key in selected.index}
    metrics.to_csv(OUT / "june_metrics.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(OUT / "june_class_metrics.csv", index=False, encoding="utf-8-sig")
    package = {
        "schema_version": 1, "input": ["geom"],
        "forbidden_inputs": ["rfid_id", "part_dt", "rfid_time"],
        "hypothesis": "joint fishing-ground and trajectory-behavior archetype posteriors",
        "selection": selected_record, "may_full_metrics": may_result,
        "may_diagnostics": may_diagnostics, "june_diagnostics": june_diagnostics,
        "source": snapshot,
    }
    (OUT / "model_package.json").write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Geom-only forward-time route-archetype ensemble", "",
        "Past labels define smoothed joint posteriors over fishing-ground tiles and trajectory-behavior bins. Prediction still accepts geom only.", "",
        "## June metrics", "", metrics.to_markdown(index=False, floatfmt=".6f"), "",
        "## Final class metrics", "", classes.to_markdown(index=False, floatfmt=".6f"), "",
        "## Frozen May rule", "", pd.DataFrame([selected_record]).to_markdown(index=False, floatfmt=".6f"), "",
        pd.DataFrame([may_result]).to_markdown(index=False, floatfmt=".6f"), "",
        "This is forward-time evaluation on a mostly recurring vessel population, not unseen-vessel evaluation.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(classes.to_string(index=False), flush=True)
    print(json.dumps(selected_record, indent=2), flush=True)


if __name__ == "__main__":
    main()
