"""Cached, vessel-disjoint comparison of CatBoost and ExtraTrees.

Screen on fold 1, then complete the other folds only for promising models.
The holdout is evaluated only after the CV recipe has been frozen.
"""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn

from src.alternative_models import PARAMETERS, create_model, fit_model
from src.common import OUTPUT, ROOT
from src.context_features import sha256
from src.improved_training import row_scores, save_array, save_json


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict(model: Any, matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.concatenate([
        model.predict_proba(matrix[rows[start:start + 50000]]).astype("float32")
        for start in range(0, len(rows), 50000)
    ])


def compare_predictions(truth: np.ndarray, control: np.ndarray, alternative: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"alternative_weight": weight, **row_scores(truth, (1 - weight) * control + weight * alternative)}
        for weight in [0.0, 0.5, 1.0]
    ]


def train_fold(name: str, fold: int, df: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
               matrix: np.ndarray, threads: int, cache: Path, out: Path, baseline: ModuleType) -> list[dict[str, Any]]:
    probability_path = cache / f"{name}_fold{fold}.npy"
    record_path = out / f"{name}_fold{fold}.json"
    if probability_path.exists() and record_path.exists():
        print(f"{name} fold {fold}: cached", flush=True)
        return read_json(record_path)["scores"]
    train, valid = np.flatnonzero((folds > 0) & (folds != fold)), np.flatnonzero(folds == fold)
    vessels = df.rfid_id.to_numpy()
    rows = baseline.sample_training(train, vessels, cap=256, seed=42)
    if set(vessels[rows]) & set(vessels[valid]):
        raise ValueError("Fold vessel overlap")
    print(f"{name} fold {fold}: train {len(rows):,}, evaluate {len(valid):,}", flush=True)
    started = time.perf_counter()
    model = create_model(name, threads)
    fit_model(name, model, matrix[rows], y[rows])
    probability = predict(model, matrix, valid)
    save_array(probability_path, probability)
    control = np.load(OUTPUT / "cache/improved_training/improved_v1" / f"control_fold{fold}.npy")
    scores = compare_predictions(y[valid], control, probability)
    for item in scores:
        item.update(candidate=name, fold=fold)
    save_json(record_path, {"seconds": time.perf_counter() - started, "train_rows": len(rows), "scores": scores})
    print(json.dumps(scores), flush=True)
    del model, probability
    gc.collect()
    return scores


def choose_model(records: list[dict[str, Any]], names: list[str], out: Path) -> dict[str, Any]:
    table = pd.DataFrame(records)
    table.to_csv(out / "cv_results.csv", index=False)
    complete = table.groupby("candidate").fold.nunique()
    table = table.loc[table.candidate.isin(complete.index[complete == 3])]
    if table.empty:
        chosen = {"accepted": False, "reason": "No alternative passed the pilot"}
    else:
        summary = table.groupby(["candidate", "alternative_weight"], as_index=False).agg(
            accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
            accuracy_std=("accuracy", "std"), macro_f1_std=("macro_f1", "std"),
        )
        control = summary.loc[summary.alternative_weight == 0].iloc[0]
        summary["eligible"] = (summary.accuracy > control.accuracy) & (summary.macro_f1 > control.macro_f1)
        summary["joint"] = (summary.accuracy + summary.macro_f1) / 2
        summary = summary.sort_values(["eligible", "joint"], ascending=False)
        summary.to_csv(out / "cv_summary.csv", index=False)
        best = summary.iloc[0]
        chosen = {"accepted": bool(best.eligible), "candidate": best.candidate,
                  "alternative_weight": float(best.alternative_weight),
                  "cv_accuracy": float(best.accuracy), "cv_macro_f1": float(best.macro_f1),
                  "control_accuracy": float(control.accuracy), "control_macro_f1": float(control.macro_f1)}
    chosen["candidates"] = names
    save_json(out / "selection.json", chosen)
    return chosen


def final_fit(selection: dict[str, Any], df: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
              matrix: np.ndarray, config: dict[str, Any], threads: int, out: Path, baseline: ModuleType) -> None:
    if not selection["accepted"] or (out / "metrics.json").exists():
        return
    name = selection["candidate"]
    extension = ".cbm" if name == "catboost" else ".joblib"
    model_path = out / ("model" + extension)
    train, valid = np.flatnonzero(folds > 0), np.flatnonzero(folds == 0)
    rows = baseline.sample_training(train, df.rfid_id.to_numpy(), 256, seed=42)
    if model_path.exists():
        if name == "catboost":
            model = catboost.CatBoostClassifier().load_model(str(model_path))
        else:
            model = joblib.load(model_path)
    else:
        print(f"Final {name}: {len(rows):,} training rows", flush=True)
        model = create_model(name, threads)
        fit_model(name, model, matrix[rows], y[rows])
        temporary = model_path.with_suffix(".tmp" + extension)
        if name == "catboost":
            model.save_model(str(temporary))
        else:
            joblib.dump(model, temporary, compress=3)
        temporary.replace(model_path)
    alternative = predict(model, matrix, valid)
    labels, columns = config["classes"], config["base_features"]
    # Reuse the control's completed, row-keyed full validation predictions.
    control_frame = pd.read_csv(OUTPUT / "experiments/final_v1/validation_predictions.csv")
    keys = ["source_file_id", "source_row", "rfid_id", "part_dt", "rfid_time", "fishing_code"]
    if not control_frame[keys].reset_index(drop=True).equals(df.iloc[valid][keys].reset_index(drop=True)):
        raise ValueError("Control and alternative validation rows differ")
    control = control_frame[[f"probability_{label}" for label in labels]].to_numpy()
    weight = selection["alternative_weight"]
    probability = (1 - weight) * control + weight * alternative
    vessels = df.iloc[valid].rfid_id.to_numpy()
    metrics = {name: baseline.evaluate(name, y[valid], p, vessels, labels, out)
               for name, p in [("control", control), ("alternative", alternative), ("selected", probability)]}
    predictions = df.iloc[valid][keys].copy()
    predictions["prediction"] = np.asarray(labels)[probability.argmax(axis=1)]
    for index, label in enumerate(labels):
        predictions[f"probability_{label}"] = probability[:, index]
    predictions.to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    save_json(out / "model_metadata.json", {
        "candidate": name, "model": model_path.name, "model_sha256": sha256(model_path),
        "classes": labels, "features": columns, "parameters": PARAMETERS[name],
        "selection": selection, "control_model": "outputs/experiments/final_v1/model.txt",
        "cap_per_vessel": 256, "seed": 42,
        "scope": "Previously inspected development vessel validation, not an independent final test",
    })
    save_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="architectures_v2")
    parser.add_argument("--candidates", nargs="+", choices=list(PARAMETERS), default=list(PARAMETERS))
    parser.add_argument("--stage", choices=["pilot", "cv", "final", "all"], default="pilot")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid run-id or threads")
    out, cache = OUTPUT / "experiments" / args.run_id, OUTPUT / "cache" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    source = OUTPUT / "experiments/improved_v1"
    manifest = {"parameters": PARAMETERS, "threads": args.threads, "candidates": args.candidates,
                "catboost_version": catboost.__version__, "sklearn_version": sklearn.__version__,
                "code": {name: sha256(ROOT / name) for name in [
                    "13_compare_architectures.py", "src/alternative_models.py", "11_refine_decisions.py",
                    "04_improve.py", "src/improved_training.py"]},
                "source_config": sha256(source / "config.json")}
    if (out / "config.json").exists():
        if read_json(out / "config.json") != manifest:
            raise ValueError("Configuration changed; use a new run-id")
    else:
        if any(out.iterdir()) or any(cache.iterdir()):
            raise ValueError("Run directories are not empty")
        save_json(out / "config.json", manifest)
    # Reuse the completed CatBoost pilot when only the forest settings changed.
    previous = OUTPUT / "experiments/architectures_v1"
    previous_cache = OUTPUT / "cache/architectures_v1"
    if out != previous and (previous / "config.json").exists():
        stored = read_json(previous / "config.json")
        same_protocol = (
            stored["parameters"]["catboost"] == PARAMETERS["catboost"]
            and stored["source_config"] == manifest["source_config"]
            and stored["catboost_version"] == manifest["catboost_version"]
            and stored["threads"] == args.threads
        )
        old_record = previous / "catboost_fold1.json"
        old_probability = previous_cache / "catboost_fold1.npy"
        if same_protocol and old_record.exists() and old_probability.exists():
            import shutil

            if not (cache / old_probability.name).exists():
                shutil.copyfile(old_probability, cache / old_probability.name)
                shutil.copyfile(old_record, out / old_record.name)
                save_json(out / "reused_pilot.json", {
                    "source": str(previous.relative_to(ROOT)),
                    "probability_sha256": sha256(old_probability),
                    "reason": "Only ExtraTrees settings changed; CatBoost inputs/settings unchanged",
                })
    print("Checking saved input/split integrity", flush=True)
    inputs, baseline = load_script("11_refine_decisions.py"), load_script("04_improve.py")
    df, y, folds, config = inputs.load_inputs(source)
    feature_manifest = read_json(OUTPUT / "cache/behavior_v2/features.json")
    if feature_manifest["signature"] != config["signature"] or feature_manifest["columns"] != config["base_features"]:
        raise ValueError("Feature cache differs")
    matrix = np.load(OUTPUT / "cache/behavior_v2/features.npy", mmap_mode="r")
    if matrix.shape != (len(df), len(config["base_features"])):
        raise ValueError("Feature cache shape differs")
    records, promising = [], []
    if args.stage != "final":
        for name in args.candidates:
            pilot = train_fold(name, 1, df, y, folds, matrix, args.threads, cache, out, baseline)
            records.extend(pilot)
            control = pilot[0]
            # Screening tolerates small metric tradeoffs; full CV selection does not.
            if any(item["accuracy"] >= control["accuracy"] - .005
                   and item["macro_f1"] >= control["macro_f1"] - .005
                   and item["accuracy"] + item["macro_f1"] > control["accuracy"] + control["macro_f1"]
                   for item in pilot[1:]):
                promising.append(name)
        save_json(out / "pilot.json", {"promising": promising, "scores": records})
        if args.stage == "pilot":
            print(f"Promising models: {promising}", flush=True)
            return
        for name in promising:
            for fold in [2, 3]:
                records.extend(train_fold(name, fold, df, y, folds, matrix, args.threads, cache, out, baseline))
        selection = choose_model(records, args.candidates, out)
        print(json.dumps(selection), flush=True)
    else:
        selection = read_json(out / "selection.json")
    if args.stage in {"all", "final"}:
        final_fit(selection, df, y, folds, matrix, config, args.threads, out, baseline)


if __name__ == "__main__":
    main()
