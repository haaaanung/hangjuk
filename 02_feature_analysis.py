"""STEP 3~4: 작은 함수 검산 → 전체 특징 생성 → EDA. 원본 파일은 수정하지 않는다."""
import argparse
import contextlib
import json
import time
import numpy as np
import pandas as pd
import shapely
from src.common import OUTPUT, FIGURES, METRICS, data_files, clean_metadata, draw_track, plt
from src.features import FEATURE_COLUMNS, extract_features


def verify_examples():
    sample = pd.read_csv(data_files()[0], nrows=5000)
    features = extract_features(sample.geom.to_numpy())
    # 모형 성능을 보고 선택하지 않는다. 모양과 품질이 다른 실제 항적을 고른다.
    moving = (features.total_distance_m > 1000) & (features.num_direction_changes >= 5) & (features.duplicate_ratio < .5)
    picks = {
        '직선적': features.loc[moving, 'straightness'].idxmax(),
        '방향 변화 큼': features.loc[moving, 'mean_direction_change_deg'].idxmax(),
        '넓은 범위': features.spatial_spread_m.idxmax(),
        '여러 선분': features.num_segments.idxmax(),
        '중복 좌표': features.duplicate_ratio.idxmax(),
    }
    selected = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (kind, idx) in zip(axes.flat, picks.items()):
        row, feat = sample.loc[idx], features.loc[idx]
        draw_track(ax, row.geom)
        ax.set_title(f'{kind}: 원본 {idx} / {row.fishing_code}\n'
                     f'distance={feat.total_distance_m:.1f}m, straightness={feat.straightness:.3f}\n'
                     f'turn={feat.mean_direction_change_deg:.1f}°, spread={feat.spatial_spread_m:.1f}m\n'
                     f'segments={feat.num_segments:.0f}, points={feat.num_points:.0f}, duplicate={feat.duplicate_ratio:.3f}', fontsize=10)
        lines = list(shapely.from_wkt(row.geom).geoms)
        # 독립 검산: 작은 실제 예제에서는 Python loop + 구면 코사인 법칙으로 계산한다.
        import math
        manual_distance = 0.
        for line in lines:
            for a, b in zip(line.coords[:-1], line.coords[1:]):
                lon1, lat1, lon2, lat2 = map(math.radians, [*a, *b])
                cosine = math.sin(lat1)*math.sin(lat2)+math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
                manual_distance += 6371008.8 * math.acos(min(1., max(-1., cosine))) if a != b else 0.
        assert abs(manual_distance - feat.total_distance_m) < max(1., .01 * feat.total_distance_m)
        assert feat.num_segments == len(lines)
        assert feat.num_points == sum(len(line.coords) for line in lines)
        coords = [tuple(p) for line in lines for p in line.coords]
        assert np.isclose(feat.duplicate_ratio, 1 - len(set(coords)) / len(coords))
        selected.append({'case': kind, 'source_row': int(idx), **row.to_dict(), **feat.to_dict(),
                         'manual_distance_m': manual_distance})
    axes.flat[-1].axis('off')
    fig.tight_layout(); fig.savefig(FIGURES / 'feature_checks.png', dpi=160); plt.close(fig)
    pd.DataFrame(selected).to_csv(OUTPUT / 'feature_check_examples.csv', index=False, encoding='utf-8-sig')
    print('실제 항적 5개 독립 거리·개수·중복 검산 통과', flush=True)


def build_table():
    start = time.perf_counter()
    destination = OUTPUT / 'feature_df.csv'
    first, audit, rejected = True, [], []
    lower, upper = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for file_id, path in enumerate(data_files(), 1):
        n = 0
        for chunk in pd.read_csv(path, chunksize=4000, low_memory=False):
            meta, valid = clean_metadata(chunk)
            features = extract_features(chunk.geom.to_numpy())
            features.index = chunk.index
            usable = valid & features.geometry_status.eq('ok')
            if (~usable).any():
                bad = chunk.loc[~usable, ['part_dt', 'rfid_id', 'rfid_time', 'fishing_code', 'geom']].copy()
                bad['source_file'] = path.name; bad['source_row'] = bad.index
                bad['reason'] = np.where(~valid.loc[bad.index], 'invalid_metadata_or_extra_fields', features.loc[bad.index, 'geometry_status'])
                rejected.append(bad)
            result = pd.concat([meta.loc[usable], features.loc[usable, FEATURE_COLUMNS]], axis=1)
            result.insert(0, 'source_row', result.index)
            result.insert(0, 'source_file_id', file_id)
            for col in ['part_dt', 'rfid_id', 'rfid_time']:
                result[col] = result[col].astype('int64')
            assert not np.isinf(result[FEATURE_COLUMNS].to_numpy()).any()
            assert result.straightness.dropna().between(0, 1 + 1e-9).all()
            result.to_csv(destination, index=False, mode='w' if first else 'a', header=first,
                          encoding='utf-8', float_format='%.10g')
            first = False; n += len(result)
            geoms = shapely.from_wkt(chunk.loc[usable, 'geom'].to_numpy())
            if len(geoms):
                bounds = shapely.bounds(geoms)
                lower = np.minimum(lower, bounds[:, :2].min(axis=0))
                upper = np.maximum(upper, bounds[:, 2:].max(axis=0))
            if chunk.index[-1] % 100000 == 99999:
                print(f'{path.name}: 원본 {chunk.index[-1]+1:,}행 처리 / {time.perf_counter()-start:.0f}s', flush=True)
        audit.append({'source_file_id': file_id, 'source_file': path.name, 'feature_rows': n})
        print(f'{path.name}: 특징 {n:,}행 완료', flush=True)
    if rejected:
        pd.concat(rejected).to_csv(OUTPUT / 'excluded_rows.csv', index=False, encoding='utf-8-sig')
    payload = {'files': audit, 'coordinate_bounds': [*lower.tolist(), *upper.tolist()],
               'excluded_rows': sum(len(x) for x in rejected), 'elapsed_seconds': time.perf_counter()-start,
               'crs': 'No CRS metadata; lon/lat inferred from coordinate ranges; spherical Haversine.'}
    (OUTPUT / 'feature_audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def analyze_table():
    df = pd.read_csv(OUTPUT / 'feature_df.csv', dtype={c: 'float32' for c in FEATURE_COLUMNS})
    # NaN은 정의상 계산 불가능한 세 경우에서만 허용한다.
    undefined_turn = df.num_direction_changes.eq(0)
    assert df.straightness.isna().equals(df.total_distance_m.eq(0))
    for col in ['mean_direction_change_deg', 'max_direction_change_deg', 'turn_rate_45']:
        assert df[col].isna().equals(undefined_turn)
    always_defined = [c for c in FEATURE_COLUMNS if c not in
                      ['straightness', 'mean_direction_change_deg', 'max_direction_change_deg', 'turn_rate_45']]
    assert not df[always_defined].isna().any().any()
    assert not np.isinf(df[FEATURE_COLUMNS].to_numpy()).any()
    assert (df.num_points >= 2 * df.num_segments).all()
    assert (df.turn_count_45 <= df.num_direction_changes).all()
    for col in ['straightness', 'duplicate_ratio', 'zero_step_ratio', 'turn_rate_45']:
        assert df[col].dropna().between(0, 1 + 1e-6).all()
    assert df.max_direction_change_deg.dropna().between(0, 180).all()
    assert not df.duplicated(['rfid_id', 'part_dt', 'rfid_time']).any()
    audit = json.loads((OUTPUT / 'feature_audit.json').read_text(encoding='utf-8'))
    assert len(df) == sum(f['feature_rows'] for f in audit['files'])
    print('전체 table 범위·NaN 정의·개수·키 중복 검증 통과', flush=True)
    with (OUTPUT / 'feature_check.txt').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        print('feature_df.head()\n', df.head().to_string())
        print('feature_df.shape', df.shape); df.info()
        print('feature_df.describe()\n', df[FEATURE_COLUMNS].describe().to_string())
        print('NaN\n', df[FEATURE_COLUMNS].isna().sum().to_string())
        print('inf', np.isinf(df[FEATURE_COLUMNS].to_numpy()).sum())
        print('top distances\n', df.nlargest(5, 'total_distance_m').to_string())
    df[FEATURE_COLUMNS].describe(percentiles=[.01, .5, .95, .99, .999]).to_csv(METRICS / 'feature_summary.csv')
    df[FEATURE_COLUMNS].isna().sum().to_csv(METRICS / 'feature_missing.csv')
    df.groupby('fishing_code')[FEATURE_COLUMNS].agg(['mean', 'median']).to_csv(METRICS / 'class_feature_summary.csv', encoding='utf-8-sig')
    counts = df.groupby('fishing_code').agg(rows=('rfid_id', 'size'), vessels=('rfid_id', 'nunique'))
    counts.to_csv(METRICS / 'label_counts.csv', encoding='utf-8-sig')
    counts.plot.bar(subplots=True, figsize=(9, 7), rot=0, legend=False)
    plt.tight_layout(); plt.savefig(FIGURES / 'label_distribution.png', dpi=160); plt.close('all')
    columns = ['total_distance_m', 'straightness', 'mean_direction_change_deg', 'spatial_spread_m']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, columns):
        for label, group in df.groupby('fishing_code'):
            values = group[col].dropna()
            if col.endswith('_m'):
                values = np.log1p(values)
            ax.hist(values, bins=50, density=True, histtype='step', label=label)
        ax.set_title(('log1p ' if col.endswith('_m') else '') + col)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / 'feature_distributions.png', dpi=160); plt.close(fig)
    corr = df[FEATURE_COLUMNS].corr()
    corr.to_csv(METRICS / 'feature_correlations.csv')
    high = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURE_COLUMNS) for b in FEATURE_COLUMNS[i+1:] if abs(corr.loc[a,b]) >= .9]
    pd.DataFrame(high, columns=['feature_a', 'feature_b', 'pearson_r']).to_csv(METRICS / 'high_correlations.csv', index=False)
    # 이상치 목록은 검토 자료다. validation 분포로 학습용 clipping 기준을 정하지 않는다.
    df.nlargest(20, 'max_step_m').to_csv(METRICS / 'largest_steps.csv', index=False, encoding='utf-8-sig')
    print('feature_df', df.shape, 'NaN', int(df[FEATURE_COLUMNS].isna().sum().sum()),
          'inf', int(np.isinf(df[FEATURE_COLUMNS].to_numpy()).sum()), flush=True)
    print(counts.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples-only', action='store_true')
    parser.add_argument('--eda-only', action='store_true')
    args = parser.parse_args()
    if not args.eda_only:
        verify_examples()
        if not args.examples_only:
            build_table()
    if not args.examples_only:
        analyze_table()
