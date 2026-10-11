"""Noise-tolerant geometry and causal, activity-conditioned vessel profiles."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from src.context_features import prior_day_history, sha256
from src.features import EARTH_RADIUS_M, grouped_sum

ROBUST_COLUMNS = ['smooth_distance_m', 'smooth_straightness', 'smooth_mean_turn_deg',
                  'smooth_turn_rate_45', 'smooth_direction_changes',
                  'heading_axial_concentration', 'shape_elongation', 'jitter_distance_fraction']


def robust_geometry(wkts):
    """Local tangent-plane metres, RDP tolerance 10m; never join separate parts.

    This is a scale-dependent shape descriptor, not a claim of GPS accuracy.
    Directions require steps longer than 10m. No point timestamps are invented.
    """
    geom = shapely.from_wkt(np.asarray(wkts, dtype=object))
    xy, row = shapely.get_coordinates(geom, return_index=True)
    n = len(geom)
    count = np.bincount(row, minlength=n)
    center = np.column_stack([grouped_sum(xy[:, j], row, n) / count for j in range(2)])
    local = np.radians(xy - center[row]) * EARTH_RADIUS_M
    local[:, 0] *= np.cos(np.radians(center[row, 1]))
    projected = shapely.set_coordinates(geom.copy(), local)
    raw_length = shapely.length(projected)
    simplified = shapely.simplify(projected, tolerance=10., preserve_topology=False)
    length = shapely.length(simplified)
    parts, part_rows = shapely.get_parts(simplified, return_index=True)
    points, point_parts = shapely.get_coordinates(parts, return_index=True)
    sizes = shapely.get_num_coordinates(parts)
    ends = np.cumsum(sizes) - 1
    starts = ends - sizes + 1
    chord = grouped_sum(np.linalg.norm(points[ends] - points[starts], axis=1), part_rows, n)
    connected = point_parts[1:] == point_parts[:-1]
    delta = np.diff(points, axis=0)[connected]
    step_parts = point_parts[1:][connected]
    step_rows = part_rows[step_parts]
    distances = np.linalg.norm(delta, axis=1)
    keep = distances > 10.
    bearing = np.arctan2(delta[keep, 1], delta[keep, 0])
    weights = distances[keep]
    moving_rows = step_rows[keep]
    adjacent = step_parts[keep][1:] == step_parts[keep][:-1]
    turns = np.abs((np.diff(bearing) + np.pi) % (2 * np.pi) - np.pi)[adjacent] * 180 / np.pi
    turn_rows = moving_rows[1:][adjacent]
    turn_n = np.bincount(turn_rows, minlength=n)
    wsum = grouped_sum(weights, moving_rows, n)
    c = grouped_sum(weights * np.cos(2 * bearing), moving_rows, n)
    s = grouped_sum(weights * np.sin(2 * bearing), moving_rows, n)
    xx = grouped_sum(local[:, 0] ** 2, row, n) / count
    yy = grouped_sum(local[:, 1] ** 2, row, n) / count
    xy_cov = grouped_sum(local[:, 0] * local[:, 1], row, n) / count
    eigen_gap = np.sqrt((xx - yy) ** 2 + 4 * xy_cov ** 2)

    def ratio(a, b):
        return np.divide(a, b, out=np.full(n, np.nan), where=b > 0)

    return pd.DataFrame({
        'smooth_distance_m': length,
        'smooth_straightness': ratio(chord, length),
        'smooth_mean_turn_deg': ratio(grouped_sum(turns, turn_rows, n), turn_n),
        'smooth_turn_rate_45': ratio(grouped_sum(turns >= 45, turn_rows, n), turn_n),
        'smooth_direction_changes': turn_n,
        'heading_axial_concentration': ratio(np.hypot(c, s), wsum),
        'shape_elongation': ratio(eigen_gap, xx + yy),
        'jitter_distance_fraction': np.clip(1 - ratio(length, raw_length), 0, 1),
    }).astype('float32')


def load_robust(df, paths, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    signature = {'code': sha256(Path(__file__)),
                 'keys': hashlib.sha256(df[['source_file_id', 'source_row']].to_numpy('int64').tobytes()).hexdigest(),
                 'sources': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]}
    # JSON normalizes tuples to lists.
    signature = json.loads(json.dumps(signature))
    manifest, array = cache / 'geometry.json', cache / 'geometry.npy'
    if manifest.exists() and array.exists() and json.loads(manifest.read_text()) == signature:
        return np.load(array, mmap_mode='r')
    result = np.full((len(df), len(ROBUST_COLUMNS)), np.nan, dtype='float32')
    assigned = np.zeros(len(df), dtype=bool)
    for file_id, source in enumerate(paths, 1):
        positions = np.flatnonzero(df.source_file_id.to_numpy() == file_id)
        lookup = pd.Series(positions, index=df.iloc[positions].source_row.to_numpy())
        for chunk in pd.read_csv(source, usecols=['geom', 'rfid_id', 'part_dt', 'rfid_time'],
                                 dtype={'rfid_id': 'string', 'part_dt': 'string', 'rfid_time': 'string'}, chunksize=10000):
            matched = chunk.index.intersection(lookup.index)
            if len(matched):
                positions = lookup.loc[matched].to_numpy()
                for key in ['rfid_id', 'part_dt', 'rfid_time']:
                    np.testing.assert_array_equal(pd.to_numeric(chunk.loc[matched, key]), df.iloc[positions][key])
                result[positions] = robust_geometry(chunk.loc[matched, 'geom']).to_numpy()
                assigned[positions] = True
        print(f'Robust geometry complete: {source.name}', flush=True)
    if not assigned.all() or np.isinf(result).any():
        raise ValueError('Geometry coverage/finite check failed')
    np.save(array, result)
    manifest.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    return result


def prior_window_means(df, columns, days):
    """Window [date-days, date), by actual date distance, not observation count."""
    keys = ['rfid_id', 'part_dt']
    group = df.groupby(keys, sort=True)
    sums, counts = group[columns].sum(), group[columns].count()
    dates = pd.to_datetime(sums.index.get_level_values('part_dt').astype(str)).to_numpy(dtype='datetime64[D]').astype('int64')
    values = np.full(sums.shape, np.nan, dtype='float32')
    for positions in sums.groupby(level=0, sort=False).indices.values():
        t = dates[positions]
        left = np.searchsorted(t, t - days, side='left')
        right = np.arange(len(t))
        total = np.vstack([np.zeros(len(columns)), sums.iloc[positions].to_numpy().cumsum(axis=0)])
        count = np.vstack([np.zeros(len(columns)), counts.iloc[positions].to_numpy().cumsum(axis=0)])
        denominator = count[right] - count[left]
        values[positions] = np.divide(total[right] - total[left], denominator,
                                      out=np.full(denominator.shape, np.nan), where=denominator > 0)
    indexer = sums.index.get_indexer(pd.MultiIndex.from_frame(df[keys]))
    return pd.DataFrame(values[indexer], index=df.index, columns=[f'recent{days}_{c}' for c in columns])


def behavior_context(df, locations, robust):
    """Append-only causal features. Never read fishing_code or source identifiers."""
    keys = ['rfid_id', 'part_dt']
    profile = df[keys].copy()
    current = pd.DataFrame(robust, index=df.index, columns=ROBUST_COLUMNS)
    active = (df.spatial_spread_m > 100) & (df.total_distance_m > 500)
    # Log moments and CDF bins retain skew/multimodality which a raw mean loses.
    for c in ['total_distance_m', 'spatial_spread_m', 'num_points', 'max_step_m']:
        profile[f'log_{c}'] = np.log1p(df[c])
    profile['active_fraction'] = active.astype('float32')
    profile['small_extent_fraction'] = (df.spatial_spread_m < 20).astype('float32')
    for threshold in [10, 100, 1000, 5000, 20000]:
        profile[f'distance_cdf_{threshold}'] = (df.total_distance_m <= threshold).astype('float32')
    for hour in range(0, 24, 4):
        profile[f'active_hour_{hour}'] = ((df.rfid_time >= hour) & (df.rfid_time < hour + 4)).astype('float32').where(active)
    continuous = ['straightness', 'mean_direction_change_deg', 'turn_rate_45', 'duplicate_ratio']
    for c in continuous:
        profile[f'active_{c}'] = df[c].where(active)
    for j, axis in enumerate(['lon', 'lat']):
        profile[f'active_{axis}'] = np.where(active, locations[:, j], np.nan)
        profile[f'quiet_{axis}'] = np.where(df.spatial_spread_m < 20, locations[:, j], np.nan)
    for c in ROBUST_COLUMNS:
        values = np.log1p(current[c]) if c in ['smooth_distance_m', 'smooth_direction_changes'] else current[c]
        profile[c] = values
    features = profile.columns.difference(keys, sort=False).tolist()
    print(f'Expanding activity profiles: {len(features)} variables', flush=True)
    past = prior_day_history(profile, features)
    # Bernoulli standard deviations are deterministic from their means.
    redundant = [c for c in past if c.startswith('past_std_') and any(t in c for t in ['fraction', 'cdf_', 'hour_'])]
    past = past.drop(columns=redundant + ['past_observations', 'past_observed_days', 'days_since_previous_date'])
    past.columns = ['behavior_' + c for c in past]
    recent_columns = ['log_total_distance_m', 'log_spatial_spread_m', 'active_fraction',
                      'small_extent_fraction', 'active_lon', 'active_lat', 'smooth_straightness', 'heading_axial_concentration']
    recent = [prior_window_means(profile, recent_columns, days) for days in [7, 30]]
    # One vote per observed day, preserving peaks instead of averaging all hours.
    daily_input = df[keys + ['total_distance_m', 'spatial_spread_m']].copy()
    daily_input['active'] = active.astype('float32')
    daily_input['lon'], daily_input['lat'] = locations[:, 0], locations[:, 1]
    daily = daily_input.groupby(keys).agg(
        daily_distance_sum=('total_distance_m', 'sum'), daily_distance_max=('total_distance_m', 'max'),
        daily_distance_median=('total_distance_m', 'median'), daily_spread_max=('spatial_spread_m', 'max'),
        daily_active_count=('active', 'sum'), daily_observation_count=('active', 'size'),
        daily_lon_std=('lon', 'std'), daily_lat_std=('lat', 'std')).reset_index()
    for c in ['daily_distance_sum', 'daily_distance_max', 'daily_distance_median', 'daily_spread_max']:
        daily[c] = np.log1p(daily[c])
    daily_columns = daily.columns.difference(keys, sort=False).tolist()
    daily_past = prior_day_history(daily, daily_columns).drop(columns=['past_observations', 'past_observed_days', 'days_since_previous_date'])
    daily_past.index = pd.MultiIndex.from_frame(daily[keys])
    daily_rows = daily_past.index.get_indexer(pd.MultiIndex.from_frame(df[keys]))
    day_features = pd.DataFrame(daily_past.to_numpy()[daily_rows], index=df.index, columns=daily_past.columns)
    result = pd.concat([current, past, *recent, day_features], axis=1).astype('float32')
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite behavior feature')
    return result
