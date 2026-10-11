"""Audit definitions and distributions on training vessels only."""

import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import shapely
from src.common import OUTPUT, data_files
from src.features import FEATURE_COLUMNS, extract_features
from src.behavior_features import ROBUST_COLUMNS
from src.context_features import sha256
from src.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-scale-only", action="store_true")
    parser.add_argument("--model-gap-only", action="store_true")
    args = parser.parse_args()
    out = OUTPUT / "experiments" / "feature_audit_v2"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        OUTPUT / "feature_df.csv", dtype={c: "float32" for c in FEATURE_COLUMNS}
    )
    split = pd.read_csv(OUTPUT / "vessel_split.csv")
    train_ids = set(split.loc[split.split == "train", "rfid_id"])
    if args.model_gap_only:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, f1_score

        model_dir = OUTPUT / "experiments" / "context_v1"
        meta = json.loads(
            (model_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["signature"]["input_sha256"] == sha256(OUTPUT / "feature_df.csv")
        matrix = np.load(
            OUTPUT / "cache" / "context_v1" / "features.npy", mmap_mode="r"
        )
        model = lgb.Booster(
            model_str=(model_dir / "model.txt").read_text(encoding="utf-8")
        )
        assert model.feature_name() == meta["features"]
        y = pd.Categorical(df.fishing_code, categories=meta["classes"]).codes
        rng = np.random.default_rng(71)
        result = {
            "seed": 71,
            "model": "context_v1",
            "sample_unit": "uniform random observation rows",
        }
        for part in ["train", "validation"]:
            indices = np.flatnonzero(
                df.rfid_id.isin(split.loc[split.split == part, "rfid_id"])
            )
            rows = rng.choice(indices, 50000, replace=False)
            predicted = model.predict(matrix[rows], num_threads=4).argmax(axis=1)
            result[part] = {
                "sample_rows": len(rows),
                "accuracy": accuracy_score(y[rows], predicted),
                "macro_f1": f1_score(y[rows], predicted, average="macro"),
            }
        (out / "model_generalization_gap.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.cached_scale_only:
        cache = OUTPUT / "cache" / "behavior_v2"
        signature = json.loads((cache / "geometry.json").read_text())
        assert signature["code"] == sha256(ROOT / "src" / "behavior_features.py")
        assert (
            signature["keys"]
            == hashlib.sha256(
                df[["source_file_id", "source_row"]].to_numpy("int64").tobytes()
            ).hexdigest()
        )
        expected_sources = [
            [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
            for p in data_files()
        ]
        assert signature["sources"] == expected_sources
        geometry = np.load(cache / "geometry.npy", mmap_mode="r")
        training = df.rfid_id.isin(train_ids)
        masks = {
            "small_extent": training & (df.spatial_spread_m < 20),
            "apparent_active": training
            & (df.spatial_spread_m > 100)
            & (df.total_distance_m > 500),
        }
        result = {}
        for name, mask in masks.items():
            smooth = geometry[mask, ROBUST_COLUMNS.index("smooth_distance_m")]
            turns = geometry[mask, ROBUST_COLUMNS.index("smooth_mean_turn_deg")]
            result[name] = {
                "rows": int(mask.sum()),
                "original_distance_median_m": float(
                    df.loc[mask, "total_distance_m"].median()
                ),
                "simplified_distance_median_m": float(np.median(smooth)),
                "original_mean_turn_median_deg": float(
                    df.loc[mask, "mean_direction_change_deg"].median()
                ),
                "simplified_turn_defined_fraction": float(np.isfinite(turns).mean()),
            }
        (out / "geometry_scale_audit.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
        return
    train = df[df.rfid_id.isin(train_ids)].copy()
    train["points_per_segment"] = train.num_points / train.num_segments
    train["small_extent"] = train.spatial_spread_m < 20
    train["no_turn_defined"] = train.num_direction_changes == 0
    train["mostly_repeated"] = train.duplicate_ratio > 0.5
    train["apparent_active"] = (train.spatial_spread_m > 100) & (
        train.total_distance_m > 500
    )
    summary = train.groupby("fishing_code")[
        ["small_extent", "no_turn_defined", "mostly_repeated", "apparent_active"]
    ].mean()
    summary.to_csv(out / "regime_fractions.csv", encoding="utf-8-sig")
    train.groupby("fishing_code")[FEATURE_COLUMNS + ["points_per_segment"]].quantile(
        [0.1, 0.5, 0.9, 0.99]
    ).to_csv(out / "quantiles.csv", encoding="utf-8-sig")
    # Verify source alignment and independently recompute a deterministic sample.
    sample = train.sample(n=1200, random_state=17)
    raw_stats = []
    for file_id, items in sample.groupby("source_file_id"):
        targets = set(items.source_row)
        lookup = items.set_index("source_row")
        for chunk in pd.read_csv(
            data_files()[int(file_id) - 1],
            usecols=["geom", "rfid_id", "part_dt", "rfid_time"],
            dtype={"rfid_id": "string", "part_dt": "string", "rfid_time": "string"},
            chunksize=20000,
        ):
            matched = chunk.index.intersection(targets)
            if not len(matched):
                continue
            expected = lookup.loc[matched]
            for key in ["rfid_id", "part_dt", "rfid_time"]:
                np.testing.assert_array_equal(
                    pd.to_numeric(chunk.loc[matched, key]), expected[key]
                )
            calc = extract_features(chunk.loc[matched, "geom"].to_numpy())
            for col in FEATURE_COLUMNS:
                np.testing.assert_allclose(
                    calc[col], expected[col], rtol=2e-6, atol=1e-4, equal_nan=True
                )
            geometries = shapely.from_wkt(chunk.loc[matched, "geom"].to_numpy())
            for geometry in geometries:
                parts = shapely.get_parts(geometry)
                sizes = shapely.get_num_coordinates(parts)
                xy = shapely.get_coordinates(geometry)
                raw_stats.append(
                    {
                        "parts": len(parts),
                        "two_point_parts": int((sizes == 2).sum()),
                        "vertices": len(xy),
                        "unique_vertices": len(np.unique(xy, axis=0)),
                    }
                )
            targets -= set(matched)
        print(f"Audited source {int(file_id)}: {len(items)} rows", flush=True)
    stats = pd.DataFrame(raw_stats)
    audit = {
        "sample_rows": len(stats),
        "source_alignment_and_recomputation": "passed",
        "two_point_segment_fraction": float(
            stats.two_point_parts.sum() / stats.parts.sum()
        ),
        "all_features_recomputed": FEATURE_COLUMNS,
        "training_rows": len(train),
        "small_extent_fraction": float(train.small_extent.mean()),
        "no_turn_defined_fraction": float(train.no_turn_defined.mean()),
        "apparent_active_fraction": float(train.apparent_active.mean()),
        "zero_step_fraction_median": float(train.zero_step_ratio.median()),
    }
    (out / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(), flush=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
