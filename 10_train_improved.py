"""신규 선박 분류 개선: 특징 + 그룹 CV + 앙상블 + 이전 날짜 예측 누적.

실행: python -X utf8 -u 10_train_improved.py --run-id improved_v1
설정 확인만: python -X utf8 10_train_improved.py --check-config
"""

import argparse
import gc
import importlib.util
import itertools
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from src.common import OUTPUT, ROOT
from src.context_features import sha256
from src.improved_training import (
    fit_fold, fit_model, predict, row_scores, save_array, save_json,
    save_model, select_recipe, validate_config,
)
from src.prediction_history import blend_history, prior_prediction_history
from src.training_features import feature_names


def load_behavior_pipeline():
    spec = importlib.util.spec_from_file_location("behavior", ROOT / "06_behavior_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_splits(df):
    previous = OUTPUT / "experiments" / "context_v1"
    split_path = previous / "vessel_split.csv"
    folds_path = previous / "cv_assignments.csv"
    split, folds = pd.read_csv(split_path), pd.read_csv(folds_path)
    vessels = df.rfid_id.to_numpy()
    if (
        not split.rfid_id.is_unique
        or set(split.rfid_id) != set(vessels)
        or set(split.split) != {"train", "validation"}
    ):
        raise ValueError("기존 선박 분할 명단이 데이터와 일치하지 않습니다")
    train_ids = set(split.loc[split.split == "train", "rfid_id"])
    if (
        not folds.rfid_id.is_unique or set(folds.rfid_id) != train_ids
        or set(folds.fold) != {1, 2, 3}
    ):
        raise ValueError("기존 3-fold 학습 선박 명단이 올바르지 않습니다")
    fold_by_vessel = folds.set_index("rfid_id").fold
    row_folds = df.rfid_id.map(fold_by_vessel).fillna(0).to_numpy(dtype="int8")
    return split, folds, row_folds, {
        "split_sha256": sha256(split_path), "folds_sha256": sha256(folds_path)
    }


def run_cv(behavior, df, matrix, columns, y, row_folds, config, out, cache):
    records = []
    vessels = df.rfid_id.to_numpy()
    for fold in [1, 2, 3]:
        train = np.flatnonzero((row_folds > 0) & (row_folds != fold))
        valid = np.flatnonzero(row_folds == fold)
        if set(vessels[train]) & set(vessels[valid]):
            raise ValueError("CV 선박이 겹칩니다")
        for name, candidate in config["candidates"].items():
            prefix = cache / f"{name}_fold{fold}"
            probability_path = prefix.with_suffix(".npy")
            record_path = prefix.with_suffix(".json")
            print(f"\n[CV {fold}/3] {name}", flush=True)
            if probability_path.exists() and record_path.exists():
                probability = np.load(probability_path, mmap_mode="r")
                if probability.shape != (len(valid), len(np.unique(y))):
                    raise ValueError(f"잘못된 CV 캐시: {probability_path}")
                record = read_json(record_path)
                print("  완료된 결과 재사용", flush=True)
            else:
                start = time.perf_counter()
                model, rounds, fit_count = fit_fold(
                    candidate, config, matrix, columns, train, y, vessels,
                    behavior.v1.sample_training,
                )
                probability = predict(model, candidate, matrix, columns, valid, config["threads"])
                record = {
                    "candidate": name, "fold": fold, "rounds": int(rounds),
                    "train_rows": fit_count, "valid_rows": len(valid),
                    "seconds": time.perf_counter() - start, "vessel_overlap": 0,
                    **row_scores(y[valid], probability),
                }
                save_array(probability_path, probability)
                save_json(record_path, record)
                del model
            records.append(record)
            pd.DataFrame(records).to_csv(out / "cv_results.csv", index=False)
            print(f"  Accuracy {record['accuracy']:.4f} · Macro-F1 {record['macro_f1']:.4f}", flush=True)
            del probability
            gc.collect()
    return pd.DataFrame(records)


def choose_recipe(df, y, row_folds, config, out, cache):
    names = list(config["candidates"])
    combinations = [
        members for size in range(1, len(names) + 1)
        for members in itertools.combinations(names, size)
    ]
    records = []
    for fold in [1, 2, 3]:
        rows = np.flatnonzero(row_folds == fold)
        keys = df.iloc[rows][["rfid_id", "part_dt"]]
        arrays = {
            name: np.load(cache / f"{name}_fold{fold}.npy", mmap_mode="r") for name in names
        }
        for members in combinations:
            probability = np.zeros((len(rows), len(np.unique(y))), dtype="float32")
            for name in members:
                probability += arrays[name] / len(members)
            prior, days = prior_prediction_history(keys, probability)
            for strength in config["history_strengths"]:
                adjusted = blend_history(
                    probability, prior, days, strength, config["history_warmup_days"]
                )
                records.append({
                    "members": "+".join(members), "history_strength": strength,
                    "fold": fold, **row_scores(y[rows], adjusted),
                })
        print(f"  예측 조합 평가 {fold}/3 완료", flush=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "cv_recipes.csv", index=False)
    summary = table.groupby(["members", "history_strength"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    )
    summary, selected = select_recipe(summary)
    summary.to_csv(out / "cv_summary.csv", index=False)
    selection = {
        "members": selected["members"].split("+"),
        "history_strength": float(selected["history_strength"]),
        "history_warmup_days": config["history_warmup_days"],
        "cv_accuracy": float(selected["accuracy_mean"]),
        "cv_macro_f1": float(selected["macro_f1_mean"]),
        "criterion": "Both mean CV metrics >= raw control; maximize their arithmetic mean",
        "note": "CV is used for selection, so these are selection scores, not independent test scores",
    }
    save_json(out / "selection.json", selection)
    print("\n선택 결과\n" + json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return selection


def report(out, metrics, selection):
    control, selected = metrics["control"], metrics["selected"]
    improved = all(selected[key] > control[key] for key in ["accuracy", "macro_f1"])
    lines = [
        "# 학습 결과", "",
        "두 지표 모두 이번 실행의 대조 모델보다 높습니다." if improved
        else "두 지표가 모두 대조 모델을 넘지는 못했습니다. 개선이 확인됐다고 해석하지 마세요.",
        "", "| 지표 | 이번 대조 모델 | 선택 모델 | 차이 |", "|---|---:|---:|---:|",
        f"| Accuracy | {control['accuracy']:.2%} | {selected['accuracy']:.2%} | {(selected['accuracy'] - control['accuracy']) * 100:+.2f}%p |",
        f"| Macro-F1 | {control['macro_f1']:.4f} | {selected['macro_f1']:.4f} | {selected['macro_f1'] - control['macro_f1']:+.4f} |",
        "", f"선택 모델: {', '.join(selection['members'])}",
        f"이전 날짜 예측의 최대 반영 비율: {selection['history_strength']:.0%}",
        "", "## 업종별 성능", "",
        "| 업종 | Precision | Recall | F1 | 행 수 |", "|---|---:|---:|---:|---:|",
    ]
    classes = pd.read_csv(out / "selected_classification_report.csv", index_col=0)
    for label in metrics["classes"]:
        row = classes.loc[label]
        lines.append(f"| {label} | {row.precision:.4f} | {row.recall:.4f} | {row['f1-score']:.4f} | {int(row.support):,} |")
    lines += [
        "", "## 해석", "",
        "대조 모델은 기존 behavior_sqrt 구조를 이번 학습 표본 수로 재학습한 모델입니다. 과거 저장 점수와는 구분합니다.",
        "후보·앙상블·누적 비율은 학습 선박 내부 CV로만 선택했습니다. 검증 결과로 자동 재선택하지 않습니다.",
        "검증 선박은 학습 선박과 겹치지 않습니다. 이전 날짜 예측 누적에는 정답을 사용하지 않습니다.",
        "첫날에는 이력이 없어 누적 효과가 없습니다. history_slices.csv에서 첫날과 이후 날짜를 확인하세요.",
        "같은 기간의 기존 개발 검증 세트를 사용했습니다. 미래 기간에 대한 독립 테스트 결과는 아닙니다.",
        "", "모델 재사용에는 model_metadata.json의 특징 순서·앙상블 비중·누적 규칙이 모두 필요합니다.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final(behavior, df, matrix, columns, y, labels, row_folds, config, out):
    if (out / "metrics.json").exists():
        if not (out / "RESULTS.md").exists():
            report(out, read_json(out / "metrics.json"), read_json(out / "selection.json"))
        print(f"이미 최종 평가 완료: {out / 'RESULTS.md'}", flush=True)
        return
    selection = read_json(out / "selection.json")
    cv = pd.read_csv(out / "cv_results.csv")
    expected = {(name, fold) for name in config["candidates"] for fold in [1, 2, 3]}
    if set(zip(cv.candidate, cv.fold)) != expected or len(cv) != len(expected):
        raise ValueError("모든 후보의 3-fold CV를 먼저 완료하세요")
    train, valid = np.flatnonzero(row_folds > 0), np.flatnonzero(row_folds == 0)
    vessels = df.rfid_id.to_numpy()
    fit_rows = behavior.v1.sample_training(train, vessels, config["cap_per_vessel"], config["seed"])
    needed = list(dict.fromkeys(["control", *selection["members"]]))
    probabilities, model_info = {}, []
    for name in needed:
        candidate = config["candidates"][name]
        rounds = int(np.median(cv.loc[cv.candidate == name, "rounds"]))
        model_path = out / f"model_{name}.txt"
        record_path = out / f"model_{name}.json"
        print(f"\n[최종 학습] {name} · {len(fit_rows):,}행 · {rounds}회", flush=True)
        record = {"candidate": name, "rounds": rounds, "train_rows": len(fit_rows)}
        if model_path.exists() and record_path.exists():
            if read_json(record_path) != record:
                raise ValueError("최종 모델의 저장 설정이 다릅니다")
            model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
            print("  완료된 최종 모델 재사용", flush=True)
        else:
            model = fit_model(candidate, config, matrix, columns, fit_rows, y, vessels, rounds)
            save_model(model, model_path, columns, candidate)
            save_json(record_path, record)
        probabilities[name] = predict(model, candidate, matrix, columns, valid, config["threads"])
        if name in selection["members"]:
            model_info.append({
                "file": model_path.name, "feature_set": candidate["features"],
                "features": feature_names(columns, candidate["features"]),
                "weight": 1 / len(selection["members"]), "rounds": rounds,
            })
        del model
        gc.collect()
    raw = sum(probabilities[name] / len(selection["members"]) for name in selection["members"])
    prior, days = prior_prediction_history(df.iloc[valid], raw)
    adjusted = blend_history(
        raw, prior, days, selection["history_strength"], selection["history_warmup_days"]
    )
    metrics = {
        name: behavior.v1.evaluate(name, y[valid], probability, vessels[valid], labels, out)
        for name, probability in [
            ("control", probabilities["control"]), ("selected_without_history", raw),
            ("selected", adjusted),
        ]
    }
    metrics.update(classes=labels, train_rows=len(fit_rows), valid_rows=len(valid), vessel_overlap=0)
    predictions = df.iloc[valid][[
        "source_file_id", "source_row", "rfid_id", "part_dt", "rfid_time", "fishing_code"
    ]].copy()
    predictions["prediction"] = np.asarray(labels)[adjusted.argmax(axis=1)]
    predictions["prior_observed_days"] = days
    for index, label in enumerate(labels):
        predictions[f"probability_{label}"] = adjusted[:, index]
    predictions.to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    slices = []
    for name, mask in [("first_day", days == 0), ("prior_1_to_6_days", (days >= 1) & (days < 7)), ("prior_7plus_days", days >= 7)]:
        if mask.any():
            slices.append({"slice": name, "rows": int(mask.sum()), **behavior.v1.scores(y[valid][mask], adjusted[mask], vessels[valid][mask])})
    pd.DataFrame(slices).to_csv(out / "history_slices.csv", index=False)
    save_json(out / "model_metadata.json", {
        "classes": labels, "base_features": columns, "models": model_info,
        "selection": selection,
        "prediction_policy": "Equal model probability mean, then strictly prior-day probability mean; one vote per observed day. No labels or current-day peers.",
        "history_api": "src.prediction_history.prior_prediction_history + blend_history",
        "feature_api": "src.training_features.feature_batch",
        "scope": "Previously inspected, same-period development vessel holdout",
        "probabilities_calibrated": False,
    })
    # This completion marker is written after all prediction artifacts.
    save_json(out / "metrics.json", metrics)
    report(out, metrics, selection)
    print(f"\n완료: {out / 'RESULTS.md'}", flush=True)
    print(f"Accuracy {metrics['selected']['accuracy']:.4f} · Macro-F1 {metrics['selected']['macro_f1']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="improved_v1", help="새 결과 폴더 이름")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "improved_training.json")
    parser.add_argument("--stage", choices=["prepare", "cv", "final", "all"], default="all")
    parser.add_argument("--threads", type=int, help="설정 파일의 CPU 스레드 수 덮어쓰기")
    parser.add_argument("--check-config", action="store_true", help="설정·필수 파일만 확인; 특징 생성/학습 없음")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("run-id에는 영문, 숫자, 밑줄, 하이픈만 사용하세요")
    config = read_json(args.config)
    if args.threads is not None:
        config["threads"] = args.threads
    validate_config(config)
    required = [
        OUTPUT / "feature_df.csv",
        OUTPUT / "experiments/context_v1/vessel_split.csv",
        OUTPUT / "experiments/context_v1/cv_assignments.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("필수 파일 없음:\n" + "\n".join(missing))
    if args.check_config:
        print("설정 및 필수 파일 경로 확인 완료. 학습은 실행하지 않았습니다.")
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    print("[1/4] 특징 준비: 기존 캐시가 유효하면 재사용합니다.", flush=True)
    behavior = load_behavior_pipeline()
    df, matrix, columns, signature = behavior.prepare()
    if args.stage == "prepare":
        print(f"준비 완료: {len(df):,}행, 기본 특징 {len(columns)}개", flush=True)
        return
    labels = sorted(df.fishing_code.unique())
    y = pd.Categorical(df.fishing_code, categories=labels).codes.astype("int32")
    split, folds, row_folds, split_signature = read_splits(df)
    out = OUTPUT / "experiments" / args.run_id
    cache = OUTPUT / "cache" / "improved_training" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    code_files = [
        "10_train_improved.py", "06_behavior_model.py", "04_improve.py",
        "src/improved_training.py", "src/training_features.py", "src/prediction_history.py",
    ]
    manifest = {
        "config": config, "signature": signature, **split_signature,
        "code": {name: sha256(ROOT / name) for name in code_files},
        "classes": labels, "base_features": columns,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__, "sklearn": sklearn.__version__},
    }
    manifest_path = out / "config.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("설정·코드·데이터가 변경됐습니다. 새로운 --run-id를 사용하세요.")
    else:
        if any(out.iterdir()) or any(cache.iterdir()):
            raise ValueError("기존 파일이 있는 폴더입니다. 새로운 --run-id를 사용하세요.")
        save_json(manifest_path, manifest)
    split.to_csv(out / "vessel_split.csv", index=False)
    folds.to_csv(out / "cv_assignments.csv", index=False)
    if (out / "metrics.json").exists():
        report(out, read_json(out / "metrics.json"), read_json(out / "selection.json"))
        print(f"이미 완료된 실행입니다: {out / 'RESULTS.md'}", flush=True)
        return
    if args.stage in {"cv", "all"}:
        print("\n[2/4] 선박 단위 3-fold 교차검증", flush=True)
        run_cv(behavior, df, matrix, columns, y, row_folds, config, out, cache)
        if not (out / "metrics.json").exists():
            print("\n[3/4] 앙상블·이전 날짜 누적 비율 선택", flush=True)
            choose_recipe(df, y, row_folds, config, out, cache)
    if args.stage in {"final", "all"}:
        if not (out / "selection.json").exists():
            raise ValueError("먼저 --stage cv 또는 --stage all을 실행하세요")
        print("\n[4/4] 최종 학습 및 기존 개발 검증 세트 평가", flush=True)
        run_final(behavior, df, matrix, columns, y, labels, row_folds, config, out)


if __name__ == "__main__":
    main()
