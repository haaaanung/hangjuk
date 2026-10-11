"""Paired vessel bootstrap: repeated observations are not independent ships."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.context_features import sha256


def metrics_from_confusions(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype="float64"),
        where=denominator > 0,
    )
    return tp.sum(axis=-1) / cm.sum(axis=(-1, -2)), f1.mean(axis=-1)


def paired_bootstrap(previous, current, labels, repetitions=2000, seed=42):
    keys = ["source_file_id", "source_row"]
    aligned = previous.merge(
        current,
        on=keys,
        suffixes=("_old", "_new"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not aligned._merge.eq("both").all():
        raise ValueError("Evaluation rows differ")
    for col in ["rfid_id", "fishing_code"]:
        if not aligned[f"{col}_old"].eq(aligned[f"{col}_new"]).all():
            raise ValueError(f"Evaluation {col} differs")
    vessel, unique = pd.factorize(aligned.rfid_id_old, sort=True)
    truth = pd.Categorical(aligned.fishing_code_old, categories=labels).codes
    predictions = [
        pd.Categorical(aligned[f"prediction_{suffix}"], categories=labels).codes
        for suffix in ["old", "new"]
    ]
    if (truth < 0).any() or any((p < 0).any() for p in predictions):
        raise ValueError("Unknown class")
    counts = []
    for predicted in predictions:
        cm = np.zeros((len(unique), len(labels), len(labels)), dtype="float64")
        np.add.at(cm, (vessel, truth, predicted), 1)
        counts.append(cm)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=repetitions
    )
    scores = []
    for cm in counts:
        sampled = (weights @ cm.reshape(len(unique), -1)).reshape(
            repetitions, len(labels), len(labels)
        )
        scores.append(metrics_from_confusions(sampled))
    result = {
        "seed": seed,
        "repetitions": repetitions,
        "vessels": len(unique),
        "rows": len(aligned),
        "method": "paired percentile bootstrap resampling vessels, retaining all their rows",
        "scope": "conditional uncertainty on the inspected development holdout, not independent test validation",
    }
    for i, metric in enumerate(["accuracy", "macro_f1"]):
        old = float(metrics_from_confusions(counts[0].sum(axis=0))[i])
        new = float(metrics_from_confusions(counts[1].sum(axis=0))[i])
        delta = scores[1][i] - scores[0][i]
        result[metric] = {
            "previous": old,
            "current": new,
            "delta": new - old,
            "paired_vessel_95pct_interval": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous", type=Path, default=Path("outputs/experiments/context_v1")
    )
    parser.add_argument(
        "--current", type=Path, default=Path("outputs/experiments/behavior_v2_fast")
    )
    parser.add_argument(
        "--cv-runs",
        type=Path,
        nargs="+",
        help="Compare completed CV runs without reading holdout predictions",
    )
    args = parser.parse_args()
    if args.cv_runs:
        tables, sources, signature = [], [], None
        for folder in args.cv_runs:
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if signature is None:
                signature = config["signature"]
            if (
                config["signature"] != signature
                or config["cv_cap_per_vessel"] != 128
                or config["vessel_overlap"] != 0
            ):
                raise ValueError("CV inputs/protocol differ")
            scores = pd.read_csv(folder / "cv_results.csv")
            assert scores.groupby("candidate").fold.nunique().eq(3).all()
            assert scores.groupby("candidate").valid_rows.sum().eq(3179832).all()
            table = pd.read_csv(folder / "cv_summary.csv")
            table["run"] = folder.name
            tables.append(table)
            sources.append(
                {
                    "run": folder.name,
                    "summary_sha256": sha256(folder / "cv_summary.csv"),
                }
            )
        combined = pd.concat(tables, ignore_index=True).sort_values(
            "macro_f1_mean", ascending=False
        )
        combined.to_csv(args.current / "all_candidates_cv.csv", index=False)
        selected = combined.iloc[0]
        selection = {
            "run": selected["run"],
            "candidate": selected["candidate"],
            "cv_macro_f1_mean": float(selected["macro_f1_mean"]),
            "criterion": "highest mean vessel-grouped CV macro-F1",
            "sources": sources,
            "note": "Selection uses internal CV; external holdout was previously inspected in earlier development",
        }
        (args.current / "selection_across_runs.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        print(combined.to_string(index=False), flush=True)
        return
    columns = ["source_file_id", "source_row", "rfid_id", "fishing_code", "prediction"]
    old = pd.read_csv(args.previous / "validation_predictions.csv", usecols=columns)
    new = pd.read_csv(args.current / "validation_predictions.csv", usecols=columns)
    labels = json.loads(
        (args.current / "model_metadata.json").read_text(encoding="utf-8")
    )["classes"]
    result = paired_bootstrap(old, new, labels)
    (args.current / "paired_vessel_bootstrap.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
