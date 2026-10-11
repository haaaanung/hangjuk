"""Cross-fitted geographic priors, evaluated on the same unseen-vessel folds."""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common import ROOT, OUTPUT
from src.context_features import sha256
from src.spatial_prior import cross_fit_prior

spec = importlib.util.spec_from_file_location("behavior", ROOT / "06_behavior_model.py")
behavior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(behavior)
v1 = behavior.v1

ANCHORS = [
    ("current", "center_lon", "center_lat"),
    ("quiet", "behavior_past_mean_quiet_lon", "behavior_past_mean_quiet_lat"),
]


def fit(x, y, threads):
    counts = np.bincount(y, minlength=4)
    weights = np.sqrt(len(y) / (4 * counts))
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        reg_lambda=5,
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
                f"  spatial trees {env.iteration + 1}/350, {time.perf_counter() - start:.0f}s",
                flush=True,
            )

    model.fit(x, y, sample_weight=weights[y], callbacks=[progress])
    return model, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="spatial_v3")
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
    train_ids = split.loc[split.split == "train", "rfid_id"].to_numpy()
    valid_ids = split.loc[split.split == "validation", "rfid_id"].to_numpy()
    assert not set(train_ids) & set(valid_ids)
    assert set(train_ids) | set(valid_ids) == set(df.rfid_id)
    vessels = df.rfid_id.to_numpy()
    train = np.flatnonzero(np.isin(vessels, train_ids))
    valid = np.flatnonzero(np.isin(vessels, valid_ids))
    labels = sorted(df.fishing_code.unique())
    y = pd.Categorical(df.fishing_code, categories=labels).codes.astype("int32")
    out = OUTPUT / "experiments" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "signature": signature,
        "runner_sha256": sha256(Path(__file__)),
        "encoder_sha256": sha256(ROOT / "src" / "spatial_prior.py"),
        "classes": labels,
        "base_features": columns,
        "anchors": ANCHORS,
        "scales": [0.005, 0.02, 0.1],
        "strength": 5,
        "threads": args.threads,
        "cv_cap_per_vessel": 128,
        "inner_encoding_folds": 5,
        "vessel_overlap": 0,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "training": "Group cross-fitted target encodings; no self/fold labels in training inputs",
        "evaluation": "Existing same-period vessel-disjoint development holdout",
    }
    config = json.loads(json.dumps(config))
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run configuration differs")
    else:
        v1.write_json(out / "config.json", config)
    folds = pd.read_csv(old / "cv_assignments.csv")
    assert set(folds.rfid_id) == set(train_ids) and folds.rfid_id.is_unique
    if args.stage == "cv":
        results = (
            pd.read_csv(out / "cv_results.csv").to_dict("records")
            if (out / "cv_results.csv").exists()
            else []
        )
        for fold in [1, 2, 3]:
            if any(r["fold"] == fold for r in results):
                continue
            eval_ids = folds.loc[folds.fold == fold, "rfid_id"].to_numpy()
            eval_rows = train[np.isin(vessels[train], eval_ids)]
            fit_rows = v1.sample_training(
                train[~np.isin(vessels[train], eval_ids)], vessels, 128
            )
            assert not set(vessels[fit_rows]) & set(vessels[eval_rows])
            print(
                f"Spatial CV {fold}/3: cross-fitting priors for {len(fit_rows):,} rows",
                flush=True,
            )
            x = pd.DataFrame(matrix[fit_rows], columns=columns)
            prior_x, encoder = cross_fit_prior(
                x, vessels[fit_rows], y[fit_rows], ANCHORS
            )
            x = pd.concat([x, prior_x], axis=1)
            model, seconds = fit(x, y[fit_rows], args.threads)
            del x, prior_x
            probabilities = []
            for start in range(0, len(eval_rows), 100000):
                chunk = pd.DataFrame(
                    matrix[eval_rows[start : start + 100000]], columns=columns
                )
                probabilities.append(
                    model.predict_proba(
                        pd.concat([chunk, encoder.transform(chunk)], axis=1)
                    )
                )
            probabilities = np.concatenate(probabilities)
            result = v1.evaluate(
                f"cv{fold}",
                y[eval_rows],
                probabilities,
                vessels[eval_rows],
                labels,
                out,
            )
            result.update(
                candidate="spatial_crossfit_sqrt",
                fold=fold,
                training_seconds=seconds,
                train_rows=len(fit_rows),
                valid_rows=len(eval_rows),
                vessel_overlap=0,
            )
            results.append(result)
            pd.DataFrame(results).to_csv(out / "cv_results.csv", index=False)
            print(json.dumps(result), flush=True)
            del model, probabilities, encoder
            gc.collect()
        table = pd.DataFrame(results)
        table.groupby("candidate").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
        ).to_csv(out / "cv_summary.csv")
        return
    if (
        not (out / "cv_summary.csv").exists()
        or len(pd.read_csv(out / "cv_results.csv")) != 3
    ):
        raise ValueError("Complete CV before final training")
    if (out / "metrics.json").exists():
        raise ValueError("Final metrics already exist; preserve this run")
    print(
        f"Final spatial model: group cross-fitting on {len(train):,} rows", flush=True
    )
    x = pd.DataFrame(matrix[train], columns=columns)
    encoded, encoder = cross_fit_prior(x, vessels[train], y[train], ANCHORS)
    x = pd.concat([x, encoded], axis=1)
    used = x.columns.tolist()
    model, seconds = fit(x, y[train], args.threads)
    del x, encoded
    gc.collect()
    probabilities = []
    for start in range(0, len(valid), 100000):
        chunk = pd.DataFrame(matrix[valid[start : start + 100000]], columns=columns)
        probabilities.append(
            model.predict_proba(pd.concat([chunk, encoder.transform(chunk)], axis=1))
        )
    probabilities = np.concatenate(probabilities)
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
            "training_seconds": seconds,
            "delta_accuracy": result["accuracy"] - previous["accuracy"],
            "delta_macro_f1": result["macro_f1"] - previous["macro_f1"],
        },
    )
    (out / "model.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
    encoder.save(out / "spatial_prior.npz")
    v1.write_json(
        out / "model_metadata.json",
        {
            "features": used,
            "base_features": columns,
            "classes": labels,
            "params": model.get_params(),
            "signature": signature,
            "encoder_file": "spatial_prior.npz",
        },
    )
    pd.Series(model.feature_importances_, index=used, name="gain").sort_values(
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
