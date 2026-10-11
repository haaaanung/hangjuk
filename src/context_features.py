"""Label-free context: current location and strictly earlier calendar dates.

The complete observation for the current row must be available at prediction time.
No within-hour ordering or per-point timestamps are assumed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from src.features import FEATURE_COLUMNS

VERSION = 1
GEO_COLUMNS = ['center_lon', 'center_lat']
DERIVED_COLUMNS = ['mean_step_m', 'bbox_aspect', 'distance_per_spread',
                   'points_per_segment', 'hour_sin', 'hour_cos', 'month']
HISTORY_BASE = FEATURE_COLUMNS + GEO_COLUMNS + ['hour_sin', 'hour_cos']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def coordinates(wkts):
    """Mean of all observed vertices, preserving duplicate point weighting."""
    geom = shapely.from_wkt(np.asarray(wkts, dtype=object), on_invalid='ignore')
    xy, rows = shapely.get_coordinates(geom, return_index=True)
    counts = np.bincount(rows, minlength=len(wkts))
    return np.column_stack([
        np.divide(np.bincount(rows, weights=xy[:, j], minlength=len(wkts)), counts,
                  out=np.full(len(wkts), np.nan), where=counts > 0)
        for j in range(2)
    ]).astype('float32')


def load_locations(df, paths, cache_dir):
    """Join raw geometry by exact provenance, never by implicit row order."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / 'locations.npy'
    meta_path = cache_dir / 'locations.json'
    key = {
        'version': VERSION,
        'keys_sha256': hashlib.sha256(
            df[['source_file_id', 'source_row']].to_numpy(dtype='int64').tobytes()
        ).hexdigest(),
        'sources': [{'path': str(p.resolve()), 'size': p.stat().st_size,
                     'mtime_ns': p.stat().st_mtime_ns} for p in paths],
    }
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == key:
        result = np.load(path)
        if result.shape == (len(df), 2) and np.isfinite(result).all():
            return result
    result = np.full((len(df), 2), np.nan, dtype='float32')
    for file_id, source in enumerate(paths, 1):
        positions = np.flatnonzero(df.source_file_id.to_numpy() == file_id)
        target_rows = df.iloc[positions].source_row.to_numpy()
        lookup = pd.Series(positions, index=target_rows)
        for chunk in pd.read_csv(source, usecols=['geom'], chunksize=20000):
            matched = chunk.index.intersection(lookup.index)
            if len(matched):
                result[lookup.loc[matched].to_numpy()] = coordinates(chunk.loc[matched, 'geom'])
        print(f'Location features: {source.name}, {len(positions):,} rows', flush=True)
    if not np.isfinite(result).all():
        raise ValueError('Missing/invalid coordinates for accepted feature rows')
    np.save(path, result)
    meta_path.write_text(json.dumps(key, indent=2), encoding='utf-8')
    return result


def safe_ratio(a, b):
    a, b = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    return np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)


def prior_day_history(df, columns):
    """Expanding moments from earlier dates only; independent of row ordering.

    Labels and vessel identifiers never enter the returned feature matrix.
    A held-out vessel may supply its own earlier, unlabeled observations.
    """
    keys = ['rfid_id', 'part_dt']
    group = df.groupby(keys, sort=True, observed=True)
    counts = group[columns].count().astype('float64')
    sums = group[columns].sum().astype('float64')
    squares = df[keys].join(df[columns].astype('float64').pow(2)).groupby(keys)[columns].sum()
    prior_count = counts.groupby(level=0).cumsum() - counts
    prior_sum = sums.groupby(level=0).cumsum() - sums
    prior_square = squares.groupby(level=0).cumsum() - squares
    means = prior_sum / prior_count.replace(0, np.nan)
    variances = (prior_square / prior_count.replace(0, np.nan) - means.pow(2)).clip(lower=0)
    history = pd.concat([means.add_prefix('past_mean_'),
                         np.sqrt(variances).add_prefix('past_std_')], axis=1)
    sizes = group.size()
    history['past_observations'] = sizes.groupby(level=0).cumsum() - sizes
    dates = pd.Series(pd.to_datetime(history.index.get_level_values('part_dt').astype(str)),
                      index=history.index)
    history['past_observed_days'] = dates.groupby(level=0).cumcount()
    history['days_since_previous_date'] = (dates - dates.groupby(level=0).shift()).dt.days
    # Use exact keys to map back to the input order, including interleaved vessels.
    indexer = history.index.get_indexer(pd.MultiIndex.from_frame(df[keys]))
    if (indexer < 0).any():
        raise ValueError('History alignment failed')
    return pd.DataFrame(history.to_numpy(dtype='float32')[indexer],
                        index=df.index, columns=history.columns)


def build_context(df, locations):
    if df.duplicated(['rfid_id', 'part_dt', 'rfid_time']).any():
        raise ValueError('Duplicate vessel/date/hour observations')
    base = df[FEATURE_COLUMNS].copy()
    base[GEO_COLUMNS] = locations
    base['mean_step_m'] = safe_ratio(df.total_distance_m, df.num_points - df.num_segments)
    base['bbox_aspect'] = safe_ratio(df.bbox_width_m, df.bbox_height_m)
    base['distance_per_spread'] = safe_ratio(df.total_distance_m, df.spatial_spread_m)
    base['points_per_segment'] = safe_ratio(df.num_points, df.num_segments)
    hour = 2 * np.pi * df.rfid_time.to_numpy() / 24
    base['hour_sin'], base['hour_cos'] = np.sin(hour), np.cos(hour)
    base['month'] = (df.part_dt.to_numpy() // 100) % 100
    history_input = df[['rfid_id', 'part_dt']].join(base[HISTORY_BASE])
    history = prior_day_history(history_input, HISTORY_BASE)
    result = pd.concat([base, history], axis=1).astype('float32')
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite context feature')
    return result
