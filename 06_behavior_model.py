"""Second-stage feature diagnosis and vessel-disjoint evaluation."""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common import ROOT, OUTPUT, data_files
from src.context_features import sha256
from src.behavior_features import load_robust, behavior_context

spec = importlib.util.spec_from_file_location("v1", ROOT / "04_improve.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

CANDIDATES = {
    "behavior_sqrt": {
        "weight_power": 0.5,
        "rounds": 350,
        "num_leaves": 31,
        "reg_lambda": 5,
    },
    "behavior_balanced": {
        "weight_power": 1.0,
        "rounds": 350,
        "num_leaves": 31,
        "reg_lambda": 5,
    },
    "behavior_deep_balanced": {
        "weight_power": 1.0,
        "rounds": 600,
        "num_leaves": 63,
        "reg_lambda": 10,
    },
}


def prepare():
    df, old_matrix, old_columns, signature = v1.prepare()
    cache = OUTPUT / "cache" / "behavior_v2"
    cache.mkdir(parents=True, exist_ok=True)
    robust = load_robust(df, data_files(), cache)
    signature = {
        "v1": signature,
        "behavior_code": sha256(ROOT / "src" / "behavior_features.py"),
        "geometry_manifest": sha256(cache / "geometry.json"),
    }
    manifest, array = cache / "features.json", cache / "features.npy"
    if manifest.exists() and array.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta["signature"] == signature:
            return df, np.load(array, mmap_mode="r"), meta["columns"], signature
    locations = np.load(OUTPUT / "cache" / "context_v1" / "locations.npy")
    features = behavior_context(df, locations, robust)
    columns = old_columns + features.columns.tolist()
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate feature names")
    merged = np.lib.format.open_memmap(
        array, mode="w+", dtype="float32", shape=(len(df), len(columns))
    )
    merged[:, : len(old_columns)] = old_matrix
    merged[:, len(old_columns) :] = features.to_numpy()
    merged.flush()
    v1.write_json(manifest, {"signature": signature, "columns": columns})
    del merged, features, old_matrix, locations, robust
    gc.collect()
    print(f"Prepared {len(df):,} rows, {len(columns)} features", flush=True)
    return df, np.load(array, mmap_mode="r"), columns, signature


def fit(config, matrix, columns, rows, y, threads):
    counts = np.bincount(y[rows], minlength=int(y.max()) + 1)
    if (counts == 0).any():
        raise ValueError("Training split missing class")
    weights = (len(rows) / (len(counts) * counts)) ** config["weight_power"]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=config["rounds"],
        learning_rate=0.05,
        num_leaves=config["num_leaves"],
        min_child_samples=200,
        reg_lambda=config["reg_lambda"],
        random_state=42,
        n_jobs=threads,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
    )
    start = time.perf_counter()

    def progress(env):
        if (env.iteration + 1) % 100 == 0:
            print(
                f"  trees {env.iteration + 1}/{config['rounds']}, {time.perf_counter() - start:.0f}s",
                flush=True,
            )

    x = pd.DataFrame(matrix[rows], columns=columns)
    model.fit(x, y[rows], sample_weight=weights[y[rows]], callbacks=[progress])
    return model, time.perf_counter() - start


def predict(model, matrix, columns, rows):
    outputs = []
    for start in range(0, len(rows), 100000):
        outputs.append(
            model.predict_proba(
                pd.DataFrame(matrix[rows[start : start + 100000]], columns=columns)
            )
        )
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="behavior_v2")
    parser.add_argument(
        "--stage", choices=["prepare", "cv", "final", "all"], default="all"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        Path(args.run_id).name != args.run_id
        or args.run_id in {".", ".."}
        or args.threads < 1
    ):
        parser.error("Invalid run-id or threads")
    df, matrix, columns, signature = prepare()
    if args.stage == "prepare":
        return
    old = OUTPUT / "experiments" / "context_v1"
    old_config = json.loads((old / "config.json").read_text(encoding="utf-8"))
    if (
        old_config["signature"] != signature["v1"]
        or old_config["args"]["cv_cap_per_vessel"] != 128
    ):
        raise ValueError("V1 CV results cannot be reused: input/protocol differs")
    out = OUTPUT / "experiments" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    labels = sorted(df.fishing_code.unique())
    y = pd.Categorical(df.fishing_code, categories=labels).codes.astype("int32")
    vessels = df.rfid_id.to_numpy()
    split = pd.read_csv(old / "vessel_split.csv")
    train_ids = set(split.loc[split.split == "train", "rfid_id"])
    valid_ids = set(split.loc[split.split == "validation", "rfid_id"])
    if train_ids & valid_ids or train_ids | valid_ids != set(vessels):
        raise ValueError("Vessel split invalid")
    train, valid = (
        np.flatnonzero(np.isin(vessels, list(train_ids))),
        np.flatnonzero(np.isin(vessels, list(valid_ids))),
    )
    config = {
        "signature": signature,
        "candidates": CANDIDATES,
        "classes": labels,
        "features": columns,
        "cv_cap_per_vessel": 128,
        "seed": 42,
        "runner_sha256": sha256(Path(__file__)),
        "v1_runner_sha256": sha256(ROOT / "04_improve.py"),
        "threads": args.threads,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "vessel_overlap": 0,
        "selection": "mean inner CV macro_f1; same fold vessel assignments as context_v1",
        "scope": "Previously inspected development holdout; no independent future claim",
    }
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run differs; use a new run-id")
    else:
        v1.write_json(out / "config.json", config)
    if args.stage in ["cv", "all"]:
        folds = pd.read_csv(old / "cv_assignments.csv")
        if set(folds.rfid_id) != train_ids or not folds.rfid_id.is_unique:
            raise ValueError("Invalid saved folds")
        existing = (
            pd.read_csv(out / "cv_results.csv").to_dict("records")
            if (out / "cv_results.csv").exists()
            else []
        )
        for fold in [1, 2, 3]:
            eval_ids = folds.loc[folds.fold == fold, "rfid_id"].to_numpy()
            eval_rows = train[np.isin(vessels[train], eval_ids)]
            fit_rows = v1.sample_training(
                train[~np.isin(vessels[train], eval_ids)], vessels, 128
            )
            assert not set(vessels[fit_rows]) & set(vessels[eval_rows])
            for name, candidate in CANDIDATES.items():
                if any(r["candidate"] == name and r["fold"] == fold for r in existing):
                    continue
                print(
                    f"CV {fold}/3 {name}: fit {len(fit_rows):,}, evaluate {len(eval_rows):,}",
                    flush=True,
                )
                model, seconds = fit(
                    candidate, matrix, columns, fit_rows, y, args.threads
                )
                probabilities = predict(model, matrix, columns, eval_rows)
                result = v1.evaluate(
                    f"cv{fold}_{name}",
                    y[eval_rows],
                    probabilities,
                    vessels[eval_rows],
                    labels,
                    out,
                )
                result.update(
                    candidate=name,
                    fold=fold,
                    training_seconds=seconds,
                    train_rows=len(fit_rows),
                    valid_rows=len(eval_rows),
                    vessel_overlap=0,
                )
                existing.append(result)
                pd.DataFrame(existing).to_csv(out / "cv_results.csv", index=False)
                print(json.dumps(result), flush=True)
                del model, probabilities
                gc.collect()
        control = (
            pd.read_csv(old / "cv_results.csv")
            .query('candidate == "history_sqrt"')
            .copy()
        )
        control.candidate = "previous_v1"
        table = pd.concat([pd.DataFrame(existing), control], ignore_index=True)
        summary = table.groupby("candidate").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
        )
        summary.sort_values("macro_f1_mean", ascending=False).to_csv(
            out / "cv_summary.csv"
        )
    if args.stage in ["final", "all"]:
        summary = pd.read_csv(out / "cv_summary.csv", index_col=0)
        chosen = summary.macro_f1_mean.idxmax()
        if chosen == "previous_v1":
            v1.write_json(
                out / "selection.json",
                {
                    "selected": chosen,
                    "action": "Keep previous model; new features did not improve CV",
                },
            )
            print("No CV improvement; previous model retained.", flush=True)
            return
        if (out / "metrics.json").exists():
            raise ValueError("Final evaluation already exists; do not overwrite")
        print(f"Selected {chosen}. Full fit: {len(train):,} rows.", flush=True)
        model, seconds = fit(
            CANDIDATES[chosen], matrix, columns, train, y, args.threads
        )
        probabilities = predict(model, matrix, columns, valid)
        result = v1.evaluate(
            "selected", y[valid], probabilities, vessels[valid], labels, out
        )
        old_metrics = json.loads((old / "metrics.json").read_text(encoding="utf-8"))[
            "selected"
        ]
        v1.write_json(
            out / "metrics.json",
            {
                "previous_v1": old_metrics,
                "selected": result,
                "candidate": chosen,
                "training_seconds": seconds,
                "delta_accuracy": result["accuracy"] - old_metrics["accuracy"],
                "delta_macro_f1": result["macro_f1"] - old_metrics["macro_f1"],
            },
        )
        (out / "model.txt").write_text(
            model.booster_.model_to_string(), encoding="utf-8"
        )
        v1.write_json(
            out / "model_metadata.json",
            {
                "classes": labels,
                "features": columns,
                "params": model.get_params(),
                "candidate": chosen,
                "signature": signature,
            },
        )
        pd.Series(model.feature_importances_, index=columns, name="gain").sort_values(
            ascending=False
        ).to_csv(out / "feature_importance.csv")
        predictions = df.iloc[valid][
            [
                "source_file_id",
                "source_row",
                "rfid_id",
                "part_dt",
                "rfid_time",
                "fishing_code",
            ]
        ].copy()
        predictions["prediction"] = np.asarray(labels)[probabilities.argmax(axis=1)]
        for i, label in enumerate(labels):
            predictions[f"probability_{label}"] = probabilities[:, i]
        predictions.to_csv(out / "validation_predictions.csv", index=False)
        has_past = matrix[valid, columns.index("past_observations")] > 0
        slices = []
        for name, mask in [("no_prior_date", ~has_past), ("has_prior_date", has_past)]:
            slices.append(
                {
                    "slice": name,
                    "rows": int(mask.sum()),
                    **v1.scores(
                        y[valid][mask], probabilities[mask], vessels[valid][mask]
                    ),
                }
            )
        pd.DataFrame(slices).to_csv(out / "history_slices.csv", index=False)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
