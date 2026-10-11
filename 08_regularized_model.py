"""Constrain vessel fingerprint memorization; select models/ensembles by grouped OOF."""

import argparse
import gc
import importlib.util
import itertools
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common import ROOT, OUTPUT
from src.context_features import sha256

spec = importlib.util.spec_from_file_location("behavior", ROOT / "06_behavior_model.py")
behavior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(behavior)
v1 = behavior.v1

CANDIDATES = {
    "regularized": dict(
        rounds=300, leaves=15, depth=4, bins=63, vessel_equivalent=15, l2=50
    ),
    "shallow": dict(
        rounds=400, leaves=7, depth=3, bins=31, vessel_equivalent=15, l2=100
    ),
    "moderate": dict(
        rounds=500, leaves=15, depth=4, bins=63, vessel_equivalent=5, l2=20
    ),
}


def fit(config, matrix, columns, rows, y, vessels, threads):
    counts = np.bincount(y[rows], minlength=4)
    if (counts == 0).any():
        raise ValueError("Missing class")
    class_weights = np.sqrt(len(rows) / (4 * counts))
    min_leaf = max(
        50,
        round(config["vessel_equivalent"] * len(rows) / len(np.unique(vessels[rows]))),
    )
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=config["rounds"],
        learning_rate=0.05,
        num_leaves=config["leaves"],
        max_depth=config["depth"],
        max_bin=config["bins"],
        min_child_samples=min_leaf,
        reg_lambda=config["l2"],
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
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
                f"  regularized trees {env.iteration + 1}/{config['rounds']}, {time.perf_counter() - start:.0f}s",
                flush=True,
            )

    model.fit(
        pd.DataFrame(matrix[rows], columns=columns),
        y[rows],
        sample_weight=class_weights[y[rows]],
        callbacks=[progress],
    )
    return model, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="regularized_v4")
    parser.add_argument("--stage", choices=["cv", "final"], default="cv")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (
        Path(args.run_id).name != args.run_id
        or args.run_id in {".", ".."}
        or args.threads < 1
    ):
        parser.error("Invalid run-id or threads")
    df, matrix, columns, signature = behavior.prepare()
    old = OUTPUT / "experiments" / "context_v1"
    split = pd.read_csv(old / "vessel_split.csv")
    vessels = df.rfid_id.to_numpy()
    train_ids = set(split.loc[split.split == "train", "rfid_id"])
    valid_ids = set(split.loc[split.split == "validation", "rfid_id"])
    assert not train_ids & valid_ids and train_ids | valid_ids == set(vessels)
    train = np.flatnonzero(np.isin(vessels, list(train_ids)))
    valid = np.flatnonzero(np.isin(vessels, list(valid_ids)))
    labels = sorted(df.fishing_code.unique())
    y = pd.Categorical(df.fishing_code, categories=labels).codes.astype("int32")
    folds = pd.read_csv(old / "cv_assignments.csv")
    assert folds.rfid_id.is_unique and set(folds.rfid_id) == train_ids
    out, cache = OUTPUT / "experiments" / args.run_id, OUTPUT / "cache" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    config = {
        "signature": signature,
        "runner_sha256": sha256(Path(__file__)),
        "classes": labels,
        "features": columns,
        "candidates": CANDIDATES,
        "threads": args.threads,
        "cv_cap_per_vessel": 128,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "vessel_overlap": 0,
        "criterion": "mean grouped CV macro-F1; equal probability ensembles also compared",
        "scope": "Previously inspected same-period development holdout",
    }
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Run configuration changed")
    else:
        v1.write_json(out / "config.json", config)
    if args.stage == "cv":
        results = (
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
                path = cache / f"{name}_fold{fold}.npy"
                if path.exists() and any(
                    r["candidate"] == name and r["fold"] == fold for r in results
                ):
                    continue
                print(
                    f"Regularized CV {fold}/3 {name}: {len(fit_rows):,} training rows",
                    flush=True,
                )
                model, seconds = fit(
                    candidate, matrix, columns, fit_rows, y, vessels, args.threads
                )
                probabilities = behavior.predict(model, matrix, columns, eval_rows)
                np.save(path, probabilities.astype("float32"))
                result = v1.evaluate(
                    f"cv{fold}_{name}",
                    y[eval_rows],
                    probabilities,
                    vessels[eval_rows],
                    labels,
                    out,
                )
                sample = np.random.default_rng(71).choice(
                    fit_rows, min(20000, len(fit_rows)), replace=False
                )
                training = v1.scores(
                    y[sample],
                    behavior.predict(model, matrix, columns, sample),
                    vessels[sample],
                )
                result.update(
                    candidate=name,
                    fold=fold,
                    training_seconds=seconds,
                    train_rows=len(fit_rows),
                    valid_rows=len(eval_rows),
                    vessel_overlap=0,
                    train_sample_accuracy=training["accuracy"],
                    train_sample_macro_f1=training["macro_f1"],
                )
                results.append(result)
                pd.DataFrame(results).to_csv(out / "cv_results.csv", index=False)
                print(json.dumps(result), flush=True)
                del model, probabilities
                gc.collect()
        ensemble_results = []
        for size in [2, 3]:
            for members in itertools.combinations(CANDIDATES, size):
                for fold in [1, 2, 3]:
                    eval_ids = folds.loc[folds.fold == fold, "rfid_id"].to_numpy()
                    eval_rows = train[np.isin(vessels[train], eval_ids)]
                    probabilities = (
                        sum(
                            np.load(cache / f"{name}_fold{fold}.npy")
                            for name in members
                        )
                        / size
                    )
                    result = v1.scores(y[eval_rows], probabilities, vessels[eval_rows])
                    result.update(candidate="+".join(members), fold=fold)
                    ensemble_results.append(result)
        pd.DataFrame(ensemble_results).to_csv(out / "cv_ensembles.csv", index=False)
        table = pd.concat(
            [pd.DataFrame(results), pd.DataFrame(ensemble_results)], ignore_index=True
        )
        table.groupby("candidate").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
        ).sort_values("macro_f1_mean", ascending=False).to_csv(out / "cv_summary.csv")
        return
    if (out / "metrics.json").exists():
        raise ValueError("Final evaluation already exists")
    summary = pd.read_csv(out / "cv_summary.csv", index_col=0)
    selected = summary.macro_f1_mean.idxmax()
    members = selected.split("+")
    probabilities = np.zeros((len(valid), len(labels)))
    sample = np.random.default_rng(71).choice(train, 50000, replace=False)
    train_probabilities = np.zeros((len(sample), len(labels)))
    metadata, total_seconds = [], 0.0
    for name in members:
        print(f"Final {name}: all {len(train):,} training rows", flush=True)
        model, seconds = fit(
            CANDIDATES[name], matrix, columns, train, y, vessels, args.threads
        )
        probabilities += behavior.predict(model, matrix, columns, valid) / len(members)
        train_probabilities += behavior.predict(model, matrix, columns, sample) / len(
            members
        )
        (out / f"model_{name}.txt").write_text(
            model.booster_.model_to_string(), encoding="utf-8"
        )
        pd.Series(model.feature_importances_, index=columns, name="gain").sort_values(
            ascending=False
        ).to_csv(out / f"importance_{name}.csv")
        metadata.append(
            {
                "file": f"model_{name}.txt",
                "weight": 1 / len(members),
                "params": model.get_params(),
            }
        )
        total_seconds += seconds
        del model
        gc.collect()
    result = v1.evaluate(
        "selected", y[valid], probabilities, vessels[valid], labels, out
    )
    previous = json.loads((old / "metrics.json").read_text(encoding="utf-8"))[
        "selected"
    ]
    v1.write_json(
        out / "metrics.json",
        {
            "previous_v1": previous,
            "selected": result,
            "candidate": selected,
            "training_seconds": total_seconds,
            "delta_accuracy": result["accuracy"] - previous["accuracy"],
            "delta_macro_f1": result["macro_f1"] - previous["macro_f1"],
            "train_sample_metrics": v1.scores(
                y[sample], train_probabilities, vessels[sample]
            ),
        },
    )
    v1.write_json(
        out / "model_metadata.json",
        {
            "features": columns,
            "classes": labels,
            "models": metadata,
            "signature": signature,
        },
    )
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
    slices = [
        {
            "slice": name,
            "rows": int(mask.sum()),
            **v1.scores(y[valid][mask], probabilities[mask], vessels[valid][mask]),
        }
        for name, mask in [("no_prior_date", ~has_past), ("has_prior_date", has_past)]
    ]
    pd.DataFrame(slices).to_csv(out / "history_slices.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
