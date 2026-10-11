"""Five-fold vessel evaluation of full and compact LightGBM classifiers.

The saved development holdout stays separate. Feature selection is refitted
inside each training fold. Existing three-fold scores are never mixed in.
"""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from src.common import OUTPUT, ROOT
from src.compact_features import describe_feature, vessel_folds
from src.context_features import sha256
from src.improved_training import row_scores, save_array, save_json
from src.spatial_embedding import append_embedding

VARIANTS = ["control", "compact_24", "compact_24_regularized", "compact_spatial_regularized"]
FEATURE_BUDGET = 24


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_model(matrix: np.ndarray, y: np.ndarray, rows: np.ndarray, threads: int,
              feature_names: list[str] | None = None, variant: str = "control") -> lgb.LGBMClassifier:
    counts = np.bincount(y[rows], minlength=int(y.max()) + 1)
    if (counts == 0).any():
        raise ValueError("Missing training class")
    regularized = variant.endswith("_regularized")
    weights = np.sqrt(len(rows) / (len(counts) * counts))
    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=350, learning_rate=.05,
        num_leaves=15 if regularized else 31, max_depth=4 if regularized else -1,
        min_child_samples=1000 if regularized else 200,
        reg_lambda=20 if regularized else 5, max_bin=63 if regularized else 255,
        random_state=42, n_jobs=threads, verbosity=-1,
        deterministic=True, force_col_wise=True, importance_type="gain",
    )
    features = matrix[rows]
    if feature_names is not None:
        features = pd.DataFrame(features, columns=feature_names)
    model.fit(features, y[rows], sample_weight=weights[y[rows]])
    return model


def probabilities(model: Any, matrix: np.ndarray, rows: np.ndarray, threads: int) -> np.ndarray:
    booster = model if isinstance(model, lgb.Booster) else model.booster_
    return np.concatenate([
        booster.predict(matrix[rows[start:start + 50000]], num_threads=threads).astype("float32")
        for start in range(0, len(rows), 50000)
    ])


def fit_or_load(path: Path, matrix: np.ndarray, y: np.ndarray, rows: np.ndarray,
                names: list[str], variant: str, threads: int) -> lgb.Booster:
    if path.exists():
        model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    else:
        print(f"  Fit {variant}: {len(rows):,} rows, {len(names)} features", flush=True)
        fitted = fit_model(matrix, y, rows, threads, feature_names=names, variant=variant)
        temporary = path.with_suffix(".tmp.txt")
        temporary.write_text(fitted.booster_.model_to_string(), encoding="utf-8")
        temporary.replace(path)
        model = fitted.booster_
    if model.feature_name() != names:
        raise ValueError("Checkpoint feature order differs")
    return model


def feature_ranking(model: lgb.Booster, columns: list[str], out: Path, tag: str) -> list[str]:
    """Ranking uses only the fitted training-fold model, never fold labels."""
    ranking = pd.DataFrame({"feature": columns, "training_gain": model.feature_importance(importance_type="gain")})
    ranking = ranking.sort_values("training_gain", ascending=False, kind="stable").reset_index(drop=True)
    ranking.insert(1, "meaning", ranking.feature.map(describe_feature))
    ranking["selected"] = ranking.index < FEATURE_BUDGET
    ranking.to_csv(out / f"ranking_{tag}.csv", index=False, encoding="utf-8-sig")
    return ranking.feature.head(FEATURE_BUDGET).tolist()


def evaluate_fold(fold: int, df: pd.DataFrame, y: np.ndarray, folds: np.ndarray, matrix: np.ndarray,
                  columns: list[str], out: Path, cache: Path, baseline: ModuleType,
                  threads: int) -> list[dict[str, Any]]:
    train = np.flatnonzero((folds > 0) & (folds != fold))
    valid = np.flatnonzero(folds == fold)
    vessels = df.rfid_id.to_numpy()
    if set(vessels[train]) & set(vessels[valid]):
        raise ValueError("CV vessel overlap")
    rows = baseline.sample_training(train, vessels, 256, seed=42)
    sample = np.random.default_rng(71).choice(rows, min(30000, len(rows)), replace=False)
    print(f"Fold {fold}/5: {len(np.unique(vessels[valid])):,} evaluation vessels", flush=True)
    control = fit_or_load(cache / f"control_fold{fold}.txt", matrix, y, rows, columns, "control", threads)
    selected = feature_ranking(control, columns, out, f"fold{fold}")
    compact = np.asarray(matrix[:, [columns.index(name) for name in selected]], dtype="float32")
    records = []
    for variant in VARIANTS:
        path = out / f"{variant}_fold{fold}.json"
        probability_path = cache / f"{variant}_fold{fold}.npy"
        if path.exists() and probability_path.exists():
            records.append(read_json(path))
            continue
        names, features = (columns, matrix) if variant == "control" else (selected, compact)
        if variant == "compact_spatial_regularized":
            spatial_rows = baseline.sample_training(train, vessels, 8, seed=42)
            features, spatial_names = append_embedding(
                matrix, columns, compact, spatial_rows, cache / f"spatial_fold{fold}.npz")
            names = selected + spatial_names
        started = time.perf_counter()
        model = control if variant == "control" else fit_or_load(
            cache / f"{variant}_fold{fold}.txt", features, y, rows, names, variant, threads
        )
        probability = probabilities(model, features, valid, threads)
        train_metrics = row_scores(y[sample], probabilities(model, features, sample, threads))
        result = {"candidate": variant, "fold": fold, "features": len(names),
                  "train_rows": len(rows), "valid_rows": len(valid), "vessel_overlap": 0,
                  "train_sample_accuracy": train_metrics["accuracy"], "train_sample_macro_f1": train_metrics["macro_f1"],
                  **baseline.scores(y[valid], probability, vessels[valid])}
        result["macro_f1_gap"] = train_metrics["macro_f1"] - result["macro_f1"]
        result["evaluation_and_optional_fit_seconds"] = time.perf_counter() - started
        save_array(probability_path, probability)
        save_json(path, result)
        records.append(result)
        print(f"  {variant}: acc={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}, train F1={train_metrics['macro_f1']:.4f}", flush=True)
        if variant != "control":
            del model
        if variant == "compact_spatial_regularized":
            del features
    del control, compact
    gc.collect()
    return records


def choose_model(records: list[dict[str, Any]], y: np.ndarray, folds: np.ndarray,
                 cache: Path, out: Path) -> dict[str, Any]:
    table = pd.DataFrame(records)
    if table.duplicated(["candidate", "fold"]).any() or not table.groupby("candidate").fold.nunique().eq(5).all():
        raise ValueError("Every model requires exactly five completed folds")
    table.to_csv(out / "cv_results.csv", index=False)
    summary = table.groupby("candidate").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        train_macro_f1_mean=("train_sample_macro_f1", "mean"), macro_f1_gap_mean=("macro_f1_gap", "mean"),
        features=("features", "first"),
    )
    for variant in VARIANTS:
        truth = np.concatenate([y[folds == fold] for fold in range(1, 6)])
        probability = np.concatenate([np.load(cache / f"{variant}_fold{fold}.npy") for fold in range(1, 6)])
        scores = row_scores(truth, probability)
        summary.loc[variant, "oof_accuracy"] = scores["accuracy"]
        summary.loc[variant, "oof_macro_f1"] = scores["macro_f1"]
    control = summary.loc["control"]
    summary["eligible"] = (summary.accuracy_mean > control.accuracy_mean) & (summary.macro_f1_mean > control.macro_f1_mean)
    summary["joint"] = (summary.accuracy_mean + summary.macro_f1_mean) / 2
    summary = summary.sort_values(["eligible", "joint"], ascending=False)
    summary.to_csv(out / "cv_summary.csv")
    best = summary.iloc[0]
    selection = {"accepted": bool(best.eligible), "candidate": str(summary.index[0]),
                 "cv_accuracy": float(best.accuracy_mean), "cv_macro_f1": float(best.macro_f1_mean),
                 "folds": 5, "criterion": "Both mean five-fold vessel CV metrics must improve over the 163-feature control"}
    save_json(out / "selection.json", selection)
    print(json.dumps(selection), flush=True)
    return selection


def final_evaluation(selection: dict[str, Any], df: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                     matrix: np.ndarray, columns: list[str], labels: list[str], out: Path,
                     baseline: ModuleType, threads: int) -> None:
    if not selection["accepted"] or (out / "metrics.json").exists():
        return
    train, valid = np.flatnonzero(folds > 0), np.flatnonzero(folds == 0)
    vessels = df.rfid_id.to_numpy()
    rows = baseline.sample_training(train, vessels, 256, seed=42)
    # This previously fitted control used exactly the same training vessels and cap.
    control_path = OUTPUT / "experiments/final_v1/model.txt"
    control = lgb.Booster(model_str=control_path.read_text(encoding="utf-8"))
    if control.feature_name() != columns:
        raise ValueError("Control feature order differs")
    selected = feature_ranking(control, columns, out, "final")
    compact = np.asarray(matrix[:, [columns.index(name) for name in selected]], dtype="float32")
    if selection["candidate"] == "compact_spatial_regularized":
        spatial_rows = baseline.sample_training(train, vessels, 8, seed=42)
        compact, spatial_names = append_embedding(matrix, columns, compact, spatial_rows, out / "spatial_embedding.npz")
        selected += spatial_names
    model = fit_or_load(out / "model.txt", compact, y, rows, selected, selection["candidate"], threads)
    probability = probabilities(model, compact, valid, threads)
    result = baseline.evaluate("selected", y[valid], probability, vessels[valid], labels, out)
    sample = np.random.default_rng(71).choice(rows, min(30000, len(rows)), replace=False)
    training = row_scores(y[sample], probabilities(model, compact, sample, threads))
    control_training = row_scores(y[sample], probabilities(control, matrix, sample, threads))
    control_metrics = read_json(OUTPUT / "experiments/final_v1/metrics.json")["selected"]
    adopted = all(result[key] > control_metrics[key] for key in ["accuracy", "macro_f1"])
    predictions = df.iloc[valid].copy()
    predictions["prediction"] = np.asarray(labels)[probability.argmax(axis=1)]
    for index, label in enumerate(labels):
        predictions[f"probability_{label}"] = probability[:, index]
    predictions.to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"input_order": np.arange(len(selected)), "feature": selected,
                  "meaning": [describe_feature(name) for name in selected]}).to_csv(out / "selected_features.csv", index=False, encoding="utf-8-sig")
    save_json(out / "model_metadata.json", {"classes": labels, "features": selected, "model": "model.txt",
              "sha256": sha256(out / "model.txt"), "selection": selection, "cap_per_vessel": 256,
              "feature_policy": "Gain ranking fitted within each training fold; final ranking from training vessels only",
              "spatial_embedding": "spatial_embedding.npz" if "spatial" in selection["candidate"] else None,
              "independent_test": False})
    save_json(out / "metrics.json", {"control": control_metrics, "selected": result,
              "train_sample_control": control_training, "train_sample_selected": training,
              "train_sample_rows": len(sample), "features": len(selected), "valid_rows": len(valid)})
    save_json(out / "adoption.json", {"adopted": adopted, "reason": "Both development validation metrics must improve",
              "retained_model": str(((out / "model.txt") if adopted else control_path).relative_to(ROOT))})
    print(json.dumps({"selected": result, "adopted": adopted}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="compact_5fold_v1")
    parser.add_argument("--stage", choices=["cv", "final", "all"], default="all")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid run-id or threads")
    out, cache = OUTPUT / "experiments" / args.run_id, OUTPUT / "cache" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    original_path = OUTPUT / "experiments/improved_v1/config.json"
    manifest = {"folds": 5, "variants": VARIANTS, "feature_budget": FEATURE_BUDGET, "threads": args.threads,
                "seed": 42, "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
                "source": sha256(original_path), "code": {name: sha256(ROOT / name) for name in [
                    "14_compact_model.py", "src/compact_features.py", "src/spatial_embedding.py", "11_refine_decisions.py", "04_improve.py", "src/improved_training.py"]}}
    if (out / "config.json").exists():
        if read_json(out / "config.json") != manifest:
            raise ValueError("Code/config changed; use a new run-id")
    else:
        if any(out.iterdir()) or any(cache.iterdir()):
            raise ValueError("Run folders are not empty")
        save_json(out / "config.json", manifest)
    baseline, inputs = load_script("04_improve.py"), load_script("11_refine_decisions.py")
    print("Checking data and constructing five vessel folds", flush=True)
    df, y, previous_folds, config = inputs.load_inputs(OUTPUT / "experiments/improved_v1")
    folds = vessel_folds(df.rfid_id.to_numpy(), y, np.flatnonzero(previous_folds > 0))
    assignments = pd.DataFrame({"rfid_id": df.rfid_id, "fold": folds}).drop_duplicates()
    assignments.to_csv(out / "cv_assignments.csv", index=False)
    pd.crosstab(df.fishing_code, folds).to_csv(out / "fold_class_rows.csv", encoding="utf-8-sig")
    unique = df[["rfid_id", "fishing_code"]].assign(fold=folds).drop_duplicates()
    pd.crosstab(unique.fishing_code, unique.fold).to_csv(out / "fold_class_vessels.csv", encoding="utf-8-sig")
    feature_manifest = read_json(OUTPUT / "cache/behavior_v2/features.json")
    if feature_manifest["signature"] != config["signature"] or feature_manifest["columns"] != config["base_features"]:
        raise ValueError("Feature cache changed")
    matrix = np.load(OUTPUT / "cache/behavior_v2/features.npy", mmap_mode="r")
    columns = feature_manifest["columns"]
    if matrix.shape != (len(df), len(columns)):
        raise ValueError("Feature matrix shape differs")
    if args.stage in {"cv", "all"}:
        records = []
        for fold in range(1, 6):
            records.extend(evaluate_fold(fold, df, y, folds, matrix, columns, out, cache, baseline, args.threads))
            pd.DataFrame(records).to_csv(out / "cv_results.csv", index=False)
        selection = choose_model(records, y, folds, cache, out)
    else:
        selection = read_json(out / "selection.json")
    if args.stage in {"all", "final"}:
        final_evaluation(selection, df, y, folds, matrix, columns, config["classes"], out, baseline, args.threads)


if __name__ == "__main__":
    main()
