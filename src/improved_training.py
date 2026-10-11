"""Training helpers for the user-run improvement experiment."""

import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.training_features import feature_batch, feature_names


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_array(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value)
    temporary.replace(path)


def validate_config(config):
    required = {
        "seed", "threads", "cap_per_vessel", "early_stopping_rounds",
        "early_stopping_vessel_fraction", "history_strengths", "history_warmup_days",
        "candidates",
    }
    if set(config) != required:
        raise ValueError(f"설정 키 확인: 누락={required - set(config)}, 추가={set(config) - required}")
    for key in ["threads", "early_stopping_rounds", "history_warmup_days"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key}: 1 이상의 정수 필요")
    if type(config["cap_per_vessel"]) is not int or config["cap_per_vessel"] < 0:
        raise ValueError("cap_per_vessel: 0 이상의 정수 필요")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed: 0 이상 2**32 미만 정수 필요")
    if not 0 < config["early_stopping_vessel_fraction"] < 0.5:
        raise ValueError("early_stopping_vessel_fraction: 0 초과 0.5 미만 필요")
    strengths = config["history_strengths"]
    if not strengths or 0 not in strengths or any(not 0 <= x <= 1 for x in strengths):
        raise ValueError("history_strengths: 0을 포함한 [0, 1] 범위의 목록 필요")
    candidates = config["candidates"]
    if "control" not in candidates or not 1 <= len(candidates) <= 4:
        raise ValueError("control을 포함해 후보 1~4개 필요")
    candidate_keys = {
        "features", "rounds", "early_stopping", "learning_rate", "num_leaves",
        "max_depth", "min_child_samples", "reg_lambda", "colsample_bytree",
        "subsample", "class_weight_power", "vessel_weight_power",
    }
    for name, candidate in candidates.items():
        if not name.isascii() or not name.replace("_", "").isalnum():
            raise ValueError("후보 이름에는 영문, 숫자, 밑줄만 사용하세요")
        if set(candidate) != candidate_keys:
            raise ValueError(f"{name}: 후보 설정 키를 확인하세요")
        if candidate["features"] not in {"behavior", "enriched", "profile"}:
            raise ValueError(f"{name}: 잘못된 features")
        if type(candidate["early_stopping"]) is not bool:
            raise ValueError(f"{name}: early_stopping은 true/false 필요")
        for key in ["rounds", "num_leaves", "min_child_samples"]:
            if type(candidate[key]) is not int or candidate[key] < 1:
                raise ValueError(f"{name}.{key}: 양의 정수 필요")
        if candidate["num_leaves"] < 2:
            raise ValueError(f"{name}: num_leaves는 2 이상 필요")
        depth = candidate["max_depth"]
        if type(depth) is not int or (depth != -1 and depth < 1):
            raise ValueError(f"{name}: max_depth는 -1 또는 양의 정수 필요")
        for key in ["learning_rate", "colsample_bytree", "subsample"]:
            if not 0 < candidate[key] <= 1:
                raise ValueError(f"{name}.{key}: (0, 1] 범위 필요")
        for key in ["class_weight_power", "vessel_weight_power"]:
            if not 0 <= candidate[key] <= 1:
                raise ValueError(f"{name}.{key}: [0, 1] 범위 필요")
        if not np.isfinite(candidate["reg_lambda"]) or candidate["reg_lambda"] < 0:
            raise ValueError(f"{name}: reg_lambda는 0 이상 필요")


def training_weights(y, vessels, class_power, vessel_power, classes):
    """Reduce repeated-vessel dominance, then balance effective class counts."""
    _, inverse, counts = np.unique(vessels, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype("float64") ** -vessel_power
    effective = np.bincount(y, weights=weights, minlength=classes)
    if (effective == 0).any():
        raise ValueError("학습 데이터에 모든 클래스가 있어야 합니다")
    weights *= (weights.sum() / (classes * effective[y])) ** class_power
    return weights / weights.mean()


def stopping_split(rows, vessels, y, fraction, seed):
    """Select stopping vessels from the current training fold only."""
    ids, first = np.unique(vessels[rows], return_index=True)
    fit_ids, stop_ids = train_test_split(
        ids, test_size=fraction, random_state=seed, stratify=y[rows[first]]
    )
    fit_rows = rows[np.isin(vessels[rows], fit_ids)]
    stop_rows = rows[np.isin(vessels[rows], stop_ids)]
    return fit_rows, stop_rows


def make_model(candidate, config, rounds):
    params = {
        key: candidate[key] for key in [
            "learning_rate", "num_leaves", "max_depth", "min_child_samples",
            "reg_lambda", "colsample_bytree", "subsample",
        ]
    }
    return lgb.LGBMClassifier(
        **params, objective="multiclass", n_estimators=rounds,
        subsample_freq=1, random_state=config["seed"], n_jobs=config["threads"],
        verbosity=-1, deterministic=True, force_col_wise=True, importance_type="gain",
    )


def fit_model(candidate, config, matrix, columns, rows, y, vessels, rounds, stop_rows=None):
    model = make_model(candidate, config, rounds)
    start = time.perf_counter()

    def progress(env):
        if (env.iteration + 1) % 100 == 0:
            print(f"    반복 {env.iteration + 1}/{rounds} · {time.perf_counter() - start:.0f}초", flush=True)

    callbacks, fit_options = [progress], {}
    if stop_rows is not None:
        stop_x = feature_batch(matrix, columns, stop_rows, candidate["features"])
        fit_options.update(eval_set=[(stop_x, y[stop_rows])], eval_metric="multi_logloss")
        callbacks.append(lgb.early_stopping(config["early_stopping_rounds"], verbose=False))
    model.fit(
        feature_batch(matrix, columns, rows, candidate["features"]), y[rows],
        sample_weight=training_weights(
            y[rows], vessels[rows], candidate["class_weight_power"],
            candidate["vessel_weight_power"], int(y.max()) + 1,
        ),
        callbacks=callbacks, **fit_options,
    )
    return model


def fit_fold(candidate, config, matrix, columns, rows, y, vessels, sample_training):
    """Tune rounds on an inner vessel split, then refit the whole outer fold."""
    rounds = candidate["rounds"]
    if candidate["early_stopping"]:
        fit_rows, stop_rows = stopping_split(
            rows, vessels, y, config["early_stopping_vessel_fraction"], config["seed"]
        )
        fit_rows = sample_training(fit_rows, vessels, config["cap_per_vessel"], config["seed"])
        print("  학습 선박 내부에서 조기 종료 반복 수 선택", flush=True)
        model = fit_model(
            candidate, config, matrix, columns, fit_rows, y, vessels, rounds, stop_rows
        )
        rounds = model.best_iteration_ or rounds
        del model
        gc.collect()
    fit_rows = sample_training(rows, vessels, config["cap_per_vessel"], config["seed"])
    print(f"  CV 학습: {len(fit_rows):,}행 · {rounds}회", flush=True)
    model = fit_model(candidate, config, matrix, columns, fit_rows, y, vessels, rounds)
    return model, rounds, len(fit_rows)


def predict(model, candidate, matrix, columns, rows, threads):
    output = []
    for start in range(0, len(rows), 50000):
        batch = feature_batch(matrix, columns, rows[start:start + 50000], candidate["features"])
        if isinstance(model, lgb.Booster):
            probability = model.predict(batch, num_threads=threads)
        else:
            probability = model.predict_proba(batch)
        output.append(probability.astype("float32"))
    return np.concatenate(output)


def row_scores(truth, probabilities):
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    confusion = np.bincount(
        classes * truth.astype("int64") + predicted, minlength=classes**2
    ).reshape(classes, classes)
    tp = np.diag(confusion)
    denom = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(classes), where=denom > 0)
    return {"accuracy": float(tp.sum() / confusion.sum()), "macro_f1": float(f1.mean())}


def select_recipe(summary):
    """Require both mean CV metrics to match/exceed the unsmoothed control."""
    table = summary.copy()
    control = table.loc[(table.members == "control") & (table.history_strength == 0)].iloc[0]
    table["eligible"] = (
        (table.accuracy_mean >= control.accuracy_mean - 1e-12)
        & (table.macro_f1_mean >= control.macro_f1_mean - 1e-12)
    )
    table["joint_score"] = (table.accuracy_mean + table.macro_f1_mean) / 2
    table["model_count"] = table.members.str.count(r"\+") + 1
    table = table.sort_values(
        ["eligible", "joint_score", "model_count", "history_strength"],
        ascending=[False, False, True, True], kind="stable",
    )
    return table, table.iloc[0].to_dict()


def save_model(model, path, columns, candidate):
    path = Path(path)
    temporary = path.with_suffix(".tmp.txt")
    temporary.write_text(model.booster_.model_to_string(), encoding="utf-8")
    temporary.replace(path)
    pd.Series(
        model.feature_importances_, index=feature_names(columns, candidate["features"]), name="gain"
    ).sort_values(ascending=False).to_csv(path.with_suffix(".importance.csv"), encoding="utf-8-sig")
