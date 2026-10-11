"""Reuse completed improved_v1 predictions; no base-model training.

Each OOF fold is split by vessel into calibration/evaluation halves. The base
model saw neither half. Final decision parameters are locked before holdout use.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import OUTPUT, ROOT
from src.context_features import sha256
from src.decision_adjustment import apply_class_weights, fit_class_weights
from src.improved_training import row_scores, save_array, save_json
from src.prediction_history import blend_history, prior_prediction_history


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_rule(truth: np.ndarray, raw: np.ndarray, smoothed: np.ndarray) -> dict[str, Any]:
    floor = row_scores(truth, raw)
    best = sum(floor.values()) / 2
    chosen = {"history_strength": 0.0, "class_weights": [1.0] * raw.shape[1]}
    for strength, probability in [(0.0, raw), (0.8, smoothed)]:
        weights = fit_class_weights(truth, probability)
        scores = row_scores(truth, probability * weights)
        joint = sum(scores.values()) / 2
        if all(scores[key] >= floor[key] for key in floor) and joint > best + 1e-6:
            chosen = {"history_strength": strength, "class_weights": weights.tolist()}
            best = joint
    return chosen


def apply_rule(raw: np.ndarray, smoothed: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    probability = smoothed if rule["history_strength"] else raw
    return apply_class_weights(probability, np.asarray(rule["class_weights"]))


def load_inputs(source: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    config = read_json(source / "config.json")
    if sha256(OUTPUT / "feature_df.csv") != config["signature"]["v1"]["input_sha256"]:
        raise ValueError("Feature table changed since cached OOF predictions were generated")
    keys = ["rfid_id", "part_dt", "rfid_time", "source_file_id", "source_row", "fishing_code"]
    df = pd.read_csv(OUTPUT / "feature_df.csv", usecols=keys)
    split_path, fold_path = source / "vessel_split.csv", source / "cv_assignments.csv"
    if sha256(split_path) != config["split_sha256"] or sha256(fold_path) != config["folds_sha256"]:
        raise ValueError("Saved split files changed")
    split, folds = pd.read_csv(split_path), pd.read_csv(fold_path)
    if not split.rfid_id.is_unique or set(split.rfid_id) != set(df.rfid_id):
        raise ValueError("Invalid vessel split")
    train_ids = set(split.loc[split.split == "train", "rfid_id"])
    valid_ids = set(split.loc[split.split == "validation", "rfid_id"])
    if train_ids & valid_ids or train_ids | valid_ids != set(df.rfid_id):
        raise ValueError("Holdout overlap or missing vessels")
    if not folds.rfid_id.is_unique or set(folds.rfid_id) != train_ids or set(folds.fold) != {1, 2, 3}:
        raise ValueError("Invalid CV assignments")
    if df.groupby("rfid_id").fishing_code.nunique().ne(1).any():
        raise ValueError("Vessel labels are not stable")
    if df.duplicated(["rfid_id", "part_dt", "rfid_time"]).any():
        raise ValueError("Duplicated observation keys")
    y = pd.Categorical(df.fishing_code, categories=config["classes"]).codes.astype("int32")
    if (y < 0).any():
        raise ValueError("Unknown class")
    row_folds = df.rfid_id.map(folds.set_index("rfid_id").fold).fillna(0).to_numpy("int8")
    return df, y, row_folds, config


def run_cv(df: pd.DataFrame, y: np.ndarray, row_folds: np.ndarray, cache: Path, out: Path) -> None:
    raw_parts, smooth_parts, truth_parts, records, assignments = [], [], [], [], []
    for fold in [1, 2, 3]:
        rows = np.flatnonzero(row_folds == fold)
        raw = np.load(cache / f"control_fold{fold}.npy")
        if raw.shape != (len(rows), len(np.unique(y))):
            raise ValueError("OOF shape differs")
        prior, days = prior_prediction_history(df.iloc[rows], raw)
        smoothed = blend_history(raw, prior, days, .8, 3)
        vessels, first = np.unique(df.iloc[rows].rfid_id.to_numpy(), return_index=True)
        tune_ids, evaluate_ids = train_test_split(
            vessels, test_size=.5, stratify=y[rows[first]], random_state=73
        )
        tune = np.isin(df.iloc[rows].rfid_id, tune_ids)
        evaluate = ~tune
        if set(tune_ids) & set(evaluate_ids):
            raise ValueError("Calibration/evaluation vessels overlap")
        rule = choose_rule(y[rows][tune], raw[tune], smoothed[tune])
        for name, probability in [("control", raw), ("adjusted", apply_rule(raw, smoothed, rule))]:
            records.append({"fold": fold, "method": name, **row_scores(y[rows][evaluate], probability[evaluate])})
        assignments.extend({"rfid_id": int(vessel), "fold": fold, "role": role}
                           for role, ids in [("calibration", tune_ids), ("evaluation", evaluate_ids)] for vessel in ids)
        print(f"fold {fold}: {records[-2]} -> {records[-1]}", flush=True)
        raw_parts.append(raw)
        smooth_parts.append(smoothed)
        truth_parts.append(y[rows])
    table = pd.DataFrame(records)
    summary = table.groupby("method")[["accuracy", "macro_f1"]].mean()
    table.to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame(assignments).to_csv(out / "calibration_assignments.csv", index=False)
    delta = summary.loc["adjusted"] - summary.loc["control"]
    # A small gain must transfer to unseen calibration-evaluation vessels.
    accepted = bool((delta > 0).all() and delta.mean() >= .002)
    decision: dict[str, Any] = {"accepted": accepted, "cv": summary.to_dict("index"), "delta": delta.to_dict()}
    if accepted:
        decision["rule"] = choose_rule(
            np.concatenate(truth_parts), np.concatenate(raw_parts), np.concatenate(smooth_parts)
        )
    save_json(out / "selection.json", decision)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def final_evaluation(df: pd.DataFrame, y: np.ndarray, folds: np.ndarray, config: dict[str, Any], source: Path, out: Path) -> None:
    decision = read_json(out / "selection.json")
    if not decision["accepted"]:
        print("별도 선박 평가에서 개선이 부족해 후보를 폐기합니다. 최종 검증은 실행하지 않습니다.")
        return
    if (out / "metrics.json").exists():
        print("완료된 최종 평가를 재사용합니다.")
        return
    rows = np.flatnonzero(folds == 0)
    probability_path = out / "control_probabilities.npy"
    if probability_path.exists():
        raw = np.load(probability_path)
    else:
        matrix_path = OUTPUT / "cache/behavior_v2/features.npy"
        manifest = read_json(matrix_path.with_name("features.json"))
        if manifest["signature"] != config["signature"] or manifest["columns"] != config["base_features"]:
            raise ValueError("Feature cache differs from fitted model")
        matrix = np.load(matrix_path, mmap_mode="r")
        model = lgb.Booster(model_str=(source / "model_control.txt").read_text(encoding="utf-8"))
        if model.feature_name() != manifest["columns"] or matrix.shape[0] != len(df):
            raise ValueError("Model feature order or row count differs")
        raw = np.concatenate([
            model.predict(matrix[rows[start:start + 50000]], num_threads=4).astype("float32")
            for start in range(0, len(rows), 50000)
        ])
        save_array(probability_path, raw)
    prior, days = prior_prediction_history(df.iloc[rows], raw)
    adjusted = apply_rule(raw, blend_history(raw, prior, days, .8, 3), decision["rule"])
    # Reuse the established evaluator, including vessel-equal metrics/reports.
    spec = importlib.util.spec_from_file_location("baseline", ROOT / "04_improve.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    labels = config["classes"]
    vessels = df.iloc[rows].rfid_id.to_numpy()
    metrics = {name: baseline.evaluate(name, y[rows], probability, vessels, labels, out)
               for name, probability in [("control", raw), ("adjusted", adjusted)]}
    previous = read_json(source / "metrics.json")["control"]
    if any(not np.isclose(metrics["control"][key], previous[key], atol=1e-10, rtol=0) for key in previous):
        raise ValueError("Saved control metrics were not reproduced")
    result = df.iloc[rows].copy()
    result["prediction"] = np.asarray(labels)[adjusted.argmax(axis=1)]
    for index, label in enumerate(labels):
        result[f"probability_{label}"] = adjusted[:, index]
    result.to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    save_json(out / "model_metadata.json", {
        "model": str((source / "model_control.txt").relative_to(ROOT)),
        "classes": labels, "features": config["base_features"], "rule": decision["rule"],
        "history_warmup_days": 3, "probabilities_calibrated": False,
        "scope": "Previously inspected development vessel validation; no independent final test",
    })
    save_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="decision_v1")
    parser.add_argument("--stage", choices=["cv", "final", "all"], default="cv")
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("Use a simple ASCII run-id")
    source = OUTPUT / "experiments/improved_v1"
    cache = OUTPUT / "cache/improved_training/improved_v1"
    out = OUTPUT / "experiments" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    signature = {str(path.relative_to(ROOT)): sha256(path) for path in [
        Path(__file__), ROOT / "src/decision_adjustment.py", ROOT / "src/prediction_history.py",
        ROOT / "src/improved_training.py", ROOT / "04_improve.py",
        source / "config.json", source / "model_control.txt",
        *[cache / f"control_fold{fold}.npy" for fold in [1, 2, 3]],
    ]}
    if (out / "config.json").exists():
        if read_json(out / "config.json") != signature:
            raise ValueError("Inputs/code changed; use a new run-id")
    elif any(out.iterdir()):
        raise ValueError("Output directory is not empty")
    else:
        save_json(out / "config.json", signature)
    print("저장된 입력·분할 무결성 확인", flush=True)
    df, y, folds, config = load_inputs(source)
    if args.stage in {"cv", "all"} and not (out / "selection.json").exists():
        run_cv(df, y, folds, cache, out)
    if args.stage in {"final", "all"}:
        final_evaluation(df, y, folds, config, source, out)


if __name__ == "__main__":
    main()
