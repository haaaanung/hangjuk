"""단위: 거리 m, 각도 degree. 모든 이동·회전은 같은 선분 안에서만 계산한다."""
import numpy as np
import pandas as pd
import shapely

EARTH_RADIUS_M = 6_371_008.8
FEATURE_COLUMNS = [
    'total_distance_m', 'segment_chord_sum_m', 'straightness',
    'mean_direction_change_deg', 'max_direction_change_deg', 'turn_count_45',
    'turn_rate_45', 'total_turn_deg', 'spatial_spread_m', 'bbox_width_m',
    'bbox_height_m', 'num_segments', 'num_points', 'duplicate_ratio',
    'zero_step_ratio', 'num_direction_changes', 'max_step_m',
]


# 경도·위도 각도값을 구면 거리로 바꾼다. 위경도 가정은 데이터 점검 기록에 남긴다.
def haversine(a, b):
    a, b = np.radians(a), np.radians(b)
    delta = b - a
    h = np.sin(delta[..., 1] / 2)**2
    h += np.cos(a[..., 1]) * np.cos(b[..., 1]) * np.sin(delta[..., 0] / 2)**2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def grouped_sum(values, row_index, n_rows):
    return np.bincount(row_index, weights=values, minlength=n_rows)


def grouped_max(values, row_index, n_rows):
    result = np.zeros(n_rows)
    np.maximum.at(result, row_index, values)
    return result


# 전체 관측 이동거리. 끊어진 선분 사이의 거리는 포함하지 않는다.
def calculate_total_distance(step_distances, step_rows, n_rows):
    return grouped_sum(step_distances, step_rows, n_rows)


# 각 선분의 시작·끝 직선거리를 합산한다. 전체 행의 양 끝을 잇지 않는다.
def calculate_straight_distance(coordinates, starts, ends, segment_rows, n_rows):
    return grouped_sum(haversine(coordinates[starts], coordinates[ends]), segment_rows, n_rows)


# 거리 가중 선분 직선성. 관측 이동이 전혀 없으면 정의할 수 없어 NaN이다.
def calculate_straightness(chord_sum, total_distance):
    return np.divide(chord_sum, total_distance, out=np.full_like(total_distance, np.nan),
                     where=total_distance > 0)


# 길이 0인 이동은 방향이 없으므로 제외한 후 같은 선분의 인접 방향만 비교한다.
def calculate_direction_changes(a, b, distances, segment_ids, row_ids):
    moving = distances > 0
    a, b = np.radians(a[moving]), np.radians(b[moving])
    segments, rows = segment_ids[moving], row_ids[moving]
    dlon = b[:, 0] - a[:, 0]
    bearing = np.degrees(np.arctan2(np.sin(dlon) * np.cos(b[:, 1]),
        np.cos(a[:, 1]) * np.sin(b[:, 1]) - np.sin(a[:, 1]) * np.cos(b[:, 1]) * np.cos(dlon)))
    adjacent = segments[1:] == segments[:-1]
    angles = np.abs((np.diff(bearing) + 180) % 360 - 180)
    return angles[adjacent], rows[1:][adjacent]


# 45도 이상 방향 변화의 관측 횟수이며 시간당 빈도는 아니다.
def calculate_turn_count(angles, angle_rows, n_rows, threshold=45):
    return grouped_sum((angles >= threshold).astype(float), angle_rows, n_rows)


# 모든 관측점의 평균 위치에 대한 거리 RMS. 반복 관측점도 원본대로 포함한다.
def calculate_spatial_spread(coordinates, point_rows, num_points, n_rows):
    center = np.column_stack([grouped_sum(coordinates[:, j], point_rows, n_rows) /
                              np.maximum(num_points, 1) for j in range(2)])
    square_distances = haversine(coordinates, center[point_rows])**2
    return np.sqrt(grouped_sum(square_distances, point_rows, n_rows) / np.maximum(num_points, 1))


# bbox의 중간 위도에서 동서 폭, 중간 경도에서 남북 높이를 거리로 계산한다.
def calculate_bounding_box(geometries):
    bounds = shapely.bounds(geometries)
    middle = (bounds[:, :2] + bounds[:, 2:]) / 2
    west = np.column_stack([bounds[:, 0], middle[:, 1]])
    east = np.column_stack([bounds[:, 2], middle[:, 1]])
    south = np.column_stack([middle[:, 0], bounds[:, 1]])
    north = np.column_stack([middle[:, 0], bounds[:, 3]])
    return haversine(west, east), haversine(south, north)


# 한 행 내 정확히 같은 좌표의 반복 비율. 선분 경계의 반복도 품질 지표로 센다.
def calculate_duplicate_ratio(coordinates, point_rows, num_points, n_rows):
    keys = np.empty(len(coordinates), dtype=[('row', 'i8'), ('x', 'f8'), ('y', 'f8')])
    keys['row'], keys['x'], keys['y'] = point_rows, coordinates[:, 0], coordinates[:, 1]
    unique = np.unique(keys)
    counts = np.bincount(unique['row'], minlength=n_rows)
    return 1 - counts / np.maximum(num_points, 1)


def extract_features(wkts):
    """메모리 절약을 위해 한 chunk씩 계산한다. 배열의 row/segment 인덱스가 경계를 보존한다."""
    n = len(wkts)
    values = np.asarray(wkts, dtype=object).copy()
    values[pd.isna(values)] = None
    geometries = shapely.from_wkt(values, on_invalid='ignore')
    types = shapely.get_type_id(geometries)
    bounds = shapely.bounds(geometries)
    valid = np.isin(types, [1, 5]) & ~shapely.is_empty(geometries)
    valid &= np.isfinite(bounds).all(axis=1) & (bounds[:, 0] >= -180) & (bounds[:, 2] <= 180)
    valid &= (bounds[:, 1] >= -90) & (bounds[:, 3] <= 90)
    valid &= shapely.get_coordinate_dimension(geometries) == 2
    status = np.where(valid, 'ok', 'invalid_geometry')
    status[pd.isna(wkts)] = 'missing_geometry'
    geometries = geometries.copy()
    geometries[~valid] = None
    lines, segment_rows = shapely.get_parts(geometries, return_index=True)
    coordinates, point_segments = shapely.get_coordinates(lines, return_index=True)
    point_rows = segment_rows[point_segments]
    num_points = np.bincount(point_rows, minlength=n)
    num_segments = np.bincount(segment_rows, minlength=n)
    lengths = shapely.get_num_coordinates(lines)
    ends = np.cumsum(lengths) - 1
    starts = ends - lengths + 1
    connected = point_segments[1:] == point_segments[:-1]
    a, b = coordinates[:-1][connected], coordinates[1:][connected]
    step_segments = point_segments[1:][connected]
    step_rows = segment_rows[step_segments]
    distances = haversine(a, b)
    total = calculate_total_distance(distances, step_rows, n)
    chord = calculate_straight_distance(coordinates, starts, ends, segment_rows, n)
    angles, angle_rows = calculate_direction_changes(a, b, distances, step_segments, step_rows)
    angle_count = np.bincount(angle_rows, minlength=n)
    turn_count = calculate_turn_count(angles, angle_rows, n)
    turn_sum = grouped_sum(angles, angle_rows, n)
    mean_angle = np.divide(turn_sum, angle_count, out=np.full(n, np.nan), where=angle_count > 0)
    max_angle = grouped_max(angles, angle_rows, n)
    max_angle[angle_count == 0] = np.nan
    width, height = calculate_bounding_box(geometries)
    step_count = np.bincount(step_rows, minlength=n)
    result = pd.DataFrame({
        'total_distance_m': total, 'segment_chord_sum_m': chord,
        'straightness': calculate_straightness(chord, total),
        'mean_direction_change_deg': mean_angle, 'max_direction_change_deg': max_angle,
        'turn_count_45': turn_count,
        'turn_rate_45': np.divide(turn_count, angle_count, out=np.full(n, np.nan), where=angle_count > 0),
        'total_turn_deg': turn_sum,
        'spatial_spread_m': calculate_spatial_spread(coordinates, point_rows, num_points, n),
        'bbox_width_m': width, 'bbox_height_m': height,
        'num_segments': num_segments, 'num_points': num_points,
        'duplicate_ratio': calculate_duplicate_ratio(coordinates, point_rows, num_points, n),
        'zero_step_ratio': grouped_sum((distances == 0).astype(float), step_rows, n) / np.maximum(step_count, 1),
        'num_direction_changes': angle_count,
        'max_step_m': grouped_max(distances, step_rows, n),
    })
    result.loc[~valid, FEATURE_COLUMNS] = np.nan
    result['geometry_status'] = status
    return result
