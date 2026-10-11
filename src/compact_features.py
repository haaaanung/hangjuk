"""Fold-local redundancy grouping and held-out-vessel feature ranking."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.improved_training import row_scores


def vessel_folds(vessels: np.ndarray, truth: np.ndarray, train_rows: np.ndarray,
                 n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Stratify unique training vessels; reserve fold zero for the holdout."""
    table = pd.DataFrame({"vessel": vessels, "label": truth})
    if table.groupby("vessel").label.nunique().gt(1).any():
        raise ValueError("Each vessel must have one stable class")
    ids, first = np.unique(vessels[train_rows], return_index=True)
    if not np.array_equal(np.sort(train_rows), np.flatnonzero(np.isin(vessels, ids))):
        raise ValueError("Training rows must include complete vessels")
    labels = truth[train_rows[first]]
    if pd.Series(labels).value_counts().min() < n_splits:
        raise ValueError("Not enough vessels per class for stratified folds")
    assignments = pd.Series(index=ids, dtype="int8")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, evaluation) in enumerate(splitter.split(ids, labels), 1):
        assignments.loc[ids[evaluation]] = fold
    return pd.Series(vessels).map(assignments).fillna(0).to_numpy(dtype="int8")


def redundant_groups(features: np.ndarray, gains: np.ndarray, threshold: float = .98) -> list[list[int]]:
    """Group correlated columns, preferring the training-gain representative.

    Missingness patterns must also agree; conditional histories are not merged
    solely because the available values happen to be correlated.
    """
    minimum = max(3, min(500, len(features) // 10))
    correlation = pd.DataFrame(features).corr(method="spearman", min_periods=minimum).to_numpy()
    missing = np.isnan(features)
    groups: list[list[int]] = []
    for index in np.argsort(-gains, kind="stable"):
        for group in groups:
            representative = group[0]
            if (abs(correlation[index, representative]) >= threshold
                    and np.mean(missing[:, index] == missing[:, representative]) >= .98):
                group.append(int(index))
                break
        else:
            groups.append([int(index)])
    return groups


def rank_groups(model, features: np.ndarray, truth: np.ndarray, columns: list[str],
                groups: list[list[int]], limit: int = 40, repeats: int = 2, threads: int = 4) -> pd.DataFrame:
    """Measure performance loss on vessels excluded from the selector fit."""
    reference = row_scores(truth, model.booster_.predict(features, num_threads=threads))
    base_score = (reference["accuracy"] + reference["macro_f1"]) / 2
    rng = np.random.default_rng(73)
    permutations = [rng.permutation(len(features)) for _ in range(repeats)]
    gains = model.feature_importances_
    records = []
    for group in groups[:limit]:
        differences = []
        altered = features.copy()
        for permutation in permutations:
            altered[:, group] = features[np.ix_(permutation, group)]
            metrics = row_scores(truth, model.booster_.predict(altered, num_threads=threads))
            differences.append(base_score - (metrics["accuracy"] + metrics["macro_f1"]) / 2)
        records.append({
            "feature": columns[group[0]], "group_members": "|".join(columns[i] for i in group),
            "training_gain": float(gains[group[0]]),
            "permutation_drop": float(np.mean(differences)),
            "permutation_std": float(np.std(differences)),
        })
    return pd.DataFrame(records).sort_values(
        ["permutation_drop", "training_gain"], ascending=False, kind="stable"
    ).reset_index(drop=True)


def choose_features(ranking: pd.DataFrame, budget: int) -> list[str]:
    """Only retain representatives with positive held-out permutation loss."""
    selected = ranking.loc[ranking.permutation_drop > 0, "feature"].head(budget).tolist()
    if len(selected) < 4:
        raise ValueError("Too few positive feature groups to support a compact candidate")
    return selected


def describe_feature(name: str) -> str:
    """Short Korean labels for inspection; source code remains the definition."""
    if name.startswith("spatial_"):
        anchor = "현재 항적 중심" if "_current_" in name else "과거 정박 유사 관측 중심"
        return anchor + (" 좌표 결측 여부" if name.endswith("missing") else f"과 학습 공간 중심 {name.rsplit('_', 1)[1]}의 RBF 유사도")
    known = {
        "center_lon": "항적 평균 경도", "center_lat": "항적 평균 위도",
        "lon": "경도", "lat": "위도", "total_distance_m": "이동거리(m)",
        "segment_chord_sum_m": "선분별 시작·끝 직선거리 합(m)",
        "straightness": "직선성", "mean_direction_change_deg": "평균 방향 변화각",
        "max_direction_change_deg": "최대 방향 변화각", "turn_rate_45": "45도 이상 회전 비율",
        "turn_count_45": "45도 이상 회전 횟수", "total_turn_deg": "방향 변화각 합",
        "spatial_spread_m": "위치의 공간 분산 규모(m)", "bbox_width_m": "항적 영역 폭(m)",
        "bbox_height_m": "항적 영역 높이(m)", "num_segments": "선분 수", "num_points": "좌표 점 수",
        "duplicate_ratio": "중복 좌표 비율", "zero_step_ratio": "이동거리 0인 인접점 비율",
        "num_direction_changes": "유효 방향 변화 수", "max_step_m": "최대 인접점 이동거리(m)",
        "hour_sin": "시간대 sin", "hour_cos": "시간대 cos", "month": "월",
        "active_fraction": "활동 조건 충족 비율", "small_extent_fraction": "공간 분산 20m 미만 비율",
        "daily_active_count": "하루 활동 조건 관측 수", "daily_observation_count": "하루 관측 수",
        "daily_distance_sum": "하루 이동거리 합(log1p)", "daily_distance_max": "하루 최대 이동거리(log1p)",
        "daily_distance_median": "하루 이동거리 중앙값(log1p)",
        "daily_spread_max": "하루 최대 공간 분산(log1p)",
        "daily_lon_std": "하루 경도 표준편차", "daily_lat_std": "하루 위도 표준편차",
        "heading_axial_concentration": "이동 방향축 집중도", "shape_elongation": "항적의 길쭉한 정도",
        "jitter_distance_fraction": "경로 단순화로 줄어든 거리 비율",
        "smooth_distance_m": "단순화한 경로 길이", "smooth_straightness": "단순화한 경로 직선성",
        "smooth_mean_turn_deg": "단순화한 경로 평균 회전각",
        "smooth_turn_rate_45": "단순화한 경로 45도 이상 회전 비율",
        "smooth_direction_changes": "단순화한 경로 유효 회전 수",
        "past_observations": "이전 날짜까지 관측 수", "past_observed_days": "이전 관측일 수",
        "days_since_previous_date": "직전 관측일과의 날짜 간격",
        "mean_step_m": "평균 인접점 이동거리(m)", "bbox_aspect": "항적 영역 가로/세로 비율",
        "distance_per_spread": "이동거리/공간 분산", "points_per_segment": "선분당 점 수",
    }
    if name in known:
        return known[name]
    for prefix, label in [
        ("behavior_", ""), ("past_mean_", "이전 날짜까지 평균: "),
        ("past_std_", "이전 날짜까지 표준편차: "),
        ("recent7_", "이전 7일 평균: "), ("recent30_", "이전 30일 평균: "),
        ("log_", "log1p 변환: "), ("active_", "활동 조건 관측의 "),
        ("quiet_", "공간 분산 20m 미만 관측의 "),
    ]:
        if name.startswith(prefix):
            return label + describe_feature(name[len(prefix):])
    if name.startswith("hour_"):
        hour = int(name.removeprefix("hour_"))
        return f"{hour}시 이상 {hour + 4}시 미만 관측 비율"
    if name.startswith("distance_cdf_"):
        return f"이동거리 {name.removeprefix('distance_cdf_')}m 이하 비율"
    return name
