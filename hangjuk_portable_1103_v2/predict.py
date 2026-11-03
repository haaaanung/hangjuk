"""공식 1103_v2 ZIP + 포함된 모델만으로 통합 제출 엑셀 하나를 생성합니다."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from current_features import FEATURE_COLUMNS, coordinates, extract_features, robust_geometry, safe_ratio

HERE = Path(__file__).resolve().parent
CLASSES = ['연안복합', '자망', '채낚기연승낚시', '통발']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def current_rows(chunk):
    """학습 때와 같은 연산/float32 변환 순서. 다른 행이나 선박 ID는 입력에 쓰지 않습니다."""
    chunk = chunk.reset_index(drop=True)
    metadata = chunk[['part_dt', 'rfid_id', 'rfid_time']].apply(pd.to_numeric, errors='coerce')
    dates = pd.to_datetime(metadata.part_dt.astype('Int64').astype('string'), format='%Y%m%d', errors='coerce')
    valid = dates.between('2025-01-01', '2025-06-30') & (metadata.rfid_id > 0) & metadata.rfid_time.between(0, 23)
    features = extract_features(chunk.geom.to_numpy())
    good = valid & features.geometry_status.eq('ok')
    metadata = metadata.loc[good].astype('int64').reset_index(drop=True)
    if metadata.empty:
        return metadata, pd.DataFrame()
    wkts = chunk.loc[good, 'geom'].to_numpy()
    f = features.loc[good, FEATURE_COLUMNS].astype('float32').reset_index(drop=True)
    f[['center_lon', 'center_lat']] = coordinates(wkts)
    f['mean_step_m'] = safe_ratio(f.total_distance_m, f.num_points - f.num_segments)
    f['bbox_aspect'] = safe_ratio(f.bbox_width_m, f.bbox_height_m)
    f['distance_per_spread'] = safe_ratio(f.total_distance_m, f.spatial_spread_m)
    f['points_per_segment'] = safe_ratio(f.num_points, f.num_segments)
    hour = 2 * np.pi * metadata.rfid_time.to_numpy() / 24
    f['hour_sin'], f['hour_cos'] = np.sin(hour), np.cos(hour)
    f['month'] = (metadata.part_dt.to_numpy() // 100) % 100
    return metadata, pd.concat([f, robust_geometry(wkts)], axis=1).astype('float32')


def decoded_name(name):
    try:
        return name.encode('cp437').decode('cp949')
    except UnicodeError:
        return name


def scan_zip(archive, target_ids, feature_names, chunk_size):
    members = []
    for entry in archive.infolist():
        name = decoded_name(entry.filename)
        if '/테스트셋/' in name and name.lower().endswith('.csv'):
            match = re.search(r'vpass_(\d+)월', name)
            if match:
                members.append((int(match.group(1)), entry))
    members.sort(key=lambda pair: pair[0])
    if [m for m, _ in members] != list(range(1, 7)):
        raise ValueError('공식 ZIP 안에서 테스트 CSV 1~6월을 찾지 못했습니다.')
    frames = []
    for month, entry in members:
        scanned = kept = 0
        with archive.open(entry) as stream:
            for chunk in pd.read_csv(stream, encoding='utf-8-sig',
                                     usecols=['part_dt', 'rfid_id', 'rfid_time', 'geom'],
                                     chunksize=chunk_size, low_memory=False):
                scanned += len(chunk)
                selected = pd.to_numeric(chunk.rfid_id, errors='coerce').isin(target_ids)
                if selected.any():
                    meta, x = current_rows(chunk.loc[selected])
                    if not meta.empty:
                        frames.append(pd.concat([meta, x[feature_names]], axis=1))
                        kept += len(meta)
        print(f'{month}월: {scanned:,}행 읽음 / 대상 {kept:,}행 특징 추출', flush=True)
    if not frames:
        raise ValueError('제출 대상의 유효한 항적이 없습니다.')
    rows = pd.concat(frames, ignore_index=True)
    missing = set(target_ids) - set(rows.rfid_id)
    if missing:
        raise ValueError(f'항적을 찾지 못한 제출 대상 {len(missing)}척')
    if np.isinf(rows[feature_names].to_numpy()).any():
        raise ValueError('특징에 무한대가 있습니다.')
    return rows


def aggregate_a(rows, booster, feature_names, target_ids, threads):
    probabilities = booster.predict(rows[feature_names].to_numpy('float32'), num_threads=threads)
    if probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all():
        raise ValueError('Test A 예측 확률 오류')
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    active = (rows.zero_step_ratio.fillna(1).to_numpy() < .5) & (rows.spatial_spread_m.fillna(0).to_numpy() >= 200)
    vessel_ids, inv = np.unique(rows.rfid_id.to_numpy('int64'), return_inverse=True)
    count = np.bincount(inv)
    result = np.stack([np.bincount(inv, weights=probabilities[:, j]) for j in range(4)], 1) / count[:, None]
    active_count = np.bincount(inv, weights=active, minlength=len(vessel_ids))
    active_sum = np.stack([np.bincount(inv, weights=probabilities[:, j] * active,
                                    minlength=len(vessel_ids)) for j in range(4)], 1)
    good = active_count > 0
    result[good] = active_sum[good] / active_count[good, None]
    table = pd.DataFrame(result, index=vessel_ids).loc[target_ids]
    a = table.to_numpy()
    return pd.DataFrame({'rfid_id': target_ids, '예측 조업형태': np.asarray(CLASSES)[a.argmax(1)],
                         '예측조업형태 확률': a.max(1)}), int((~good).sum())


def robust_scale(a, floor):
    median = np.nanmedian(a, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(a - median), axis=0)
    return median, np.maximum(np.nan_to_num(scale, nan=0), floor)


def baseline_b(rows):
    """성능 미검증 임시 방법. Task B에 허용된 동일 선박의 관측 비교를 사용합니다."""
    names = ['total_distance_m', 'spatial_spread_m', 'mean_direction_change_deg',
             'zero_step_ratio', 'center_lon', 'center_lat']
    d = rows[['rfid_id', 'part_dt', 'rfid_time'] + names].copy()
    d['timestamp'] = (pd.to_datetime(d.part_dt.astype('int64').astype(str), format='%Y%m%d')
                      + pd.to_timedelta(d.rfid_time, unit='h'))
    d = d.groupby(['rfid_id', 'timestamp'], as_index=False)[names].mean().sort_values(['rfid_id', 'timestamp']).reset_index(drop=True)
    # 기존 캐시의 여섯 특징도 float32이며, 임시 B 연산은 float64로 수행했습니다.
    X = d[names[:4]].to_numpy(dtype=float, copy=True)
    X[:, :2] = np.log1p(np.maximum(X[:, :2], 0)); X[:, 2] /= 180
    hours = d.groupby('rfid_id').timestamp.diff().dt.total_seconds().to_numpy() / 3600
    prev = d.groupby('rfid_id')[['center_lon', 'center_lat']].shift().to_numpy(dtype=float)
    now = d[['center_lon', 'center_lat']].to_numpy(dtype=float)
    lat = np.radians((prev[:, 1] + now[:, 1]) / 2)
    dx = (now[:, 0] - prev[:, 0]) * 111320 * np.cos(lat)
    dy = (now[:, 1] - prev[:, 1]) * 110574
    shift = np.log1p(np.sqrt(dx * dx + dy * dy))
    delta = pd.DataFrame(X).groupby(d.rfid_id).diff().to_numpy()
    delta = np.column_stack([delta, shift])
    valid = (hours > 0) & (hours <= 2)
    delta[~valid] = np.nan
    floor = np.array([.2, .2, .03, .05, .2])
    gm, gs = robust_scale(delta[valid], floor) if valid.any() else (np.zeros(5), np.ones(5))
    raw = np.zeros(len(d))
    for idx in d.groupby('rfid_id', sort=False).indices.values():
        idx = np.asarray(idx); good = idx[valid[idx]]
        med, scale = robust_scale(delta[good], floor) if len(good) >= 12 else (gm, gs)
        z = np.minimum(np.nan_to_num(np.abs((delta[idx] - med) / scale), nan=0, posinf=20, neginf=0), 20)
        raw[idx] = np.sort(z, axis=1)[:, -2:].mean(axis=1)
    d['raw_anomaly'] = raw; d['anomaly_score'] = 1 - np.exp(-raw / 3)
    records = {}
    for rid, g in d.groupby('rfid_id', sort=False):
        peak = g.loc[g.raw_anomaly.idxmax()]; alarm = bool(peak.raw_anomaly >= 6)
        history = g[g.timestamp <= peak.timestamp].tail(24) if alarm else g.tail(24)
        records[int(rid)] = {'incident_time': peak.timestamp.strftime('%Y%m%d %H') if alarm else None,
                            'incident_score': float(peak.anomaly_score) if alarm else None,
                            'series': [(t.strftime('%Y%m%d %H'), float(s))
                                       for t, s in zip(history.timestamp, history.anomaly_score)]}
    return records, d


def save_workbook(template, a, records, path):
    book = load_workbook(io.BytesIO(template)); sa = book['테스트A']; sb = book['테스트B']
    expected_a = [sa.cell(i, 1).value for i in range(2, sa.max_row + 1)]
    expected_b = [sb.cell(i, 1).value for i in range(2, sb.max_row + 1)]
    if [int(i) for i in expected_a] != a.rfid_id.tolist():
        raise ValueError('Test A 순서 불일치')
    headers_a = {c.value: c.column for c in sa[1]}
    for row, item in enumerate(a.to_dict('records'), 2):
        sa.cell(row, headers_a['예측 조업형태'], item['예측 조업형태'])
        sa.cell(row, headers_a['예측조업형태 확률'], float(item['예측조업형태 확률']))
    for row in sb.iter_rows():
        for cell in row:
            cell.value = None
    n = max(3, max(len(item['series']) for item in records.values()))
    headers = ['rfid_id', '사고시각', '이상항적 점수']
    for j in range(1, n + 1):
        headers += [f'시각{j}', f'시각{j} 이상항적 점수']
    for j, value in enumerate(headers, 1):
        sb.cell(1, j, value)
    for row, rid in enumerate(expected_b, 2):
        item = records[int(rid)]
        sb.cell(row, 1, rid); sb.cell(row, 2, item['incident_time']); sb.cell(row, 3, item['incident_score'])
        for j, (timestamp, score) in enumerate(item['series']):
            sb.cell(row, 4 + j * 2, timestamp); sb.cell(row, 5 + j * 2, score)
    book.save(path)
    check = load_workbook(path, data_only=True)
    for sn, expected in [('테스트A', expected_a), ('테스트B', expected_b)]:
        if [check[sn].cell(i, 1).value for i in range(2, check[sn].max_row + 1)] != expected:
            raise ValueError(f'{sn} 저장 후 ID 순서 불일치')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', required=True, type=Path, dest='zip_path', help='공식 문제 데이터_수정_1103_v2.zip')
    ap.add_argument('--output', type=Path, default=HERE / 'output')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--chunk-size', type=int, default=10000)
    args = ap.parse_args(); start = time.perf_counter()
    if args.threads < 1 or args.chunk_size < 1:
        ap.error('threads/chunk-size는 양수여야 합니다.')
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = HERE / 'model/model_l127.txt'
    model_meta = json.loads((HERE / 'model/model_l127.json').read_text(encoding='utf-8'))
    dataset_meta = json.loads((HERE / 'dataset_manifest.json').read_text(encoding='utf-8'))
    if sha(model_path) != model_meta['model_sha256'] or model_meta['classes'] != CLASSES:
        raise ValueError('포함된 모델의 해시/클래스 검증 실패')
    feature_names = model_meta['features']
    booster = lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
    if booster.num_feature() != len(feature_names):
        raise ValueError('모델 특징 수 불일치')
    with zipfile.ZipFile(args.zip_path) as archive:
        samples = [e for e in archive.infolist() if decoded_name(e.filename).endswith('/제출샘플.xlsx')]
        if len(samples) != 1:
            raise ValueError('제출샘플.xlsx를 찾지 못했습니다.')
        template = archive.read(samples[0])
        if hashlib.sha256(template).hexdigest() != dataset_meta['template_sha256']:
            raise ValueError('1103_v2 제출샘플과 다릅니다. 데이터 버전을 확인하세요.')
        book = load_workbook(io.BytesIO(template), data_only=True)
        a_ids = [int(book['테스트A'].cell(i, 1).value) for i in range(2, 102)]
        b_ids = [int(book['테스트B'].cell(i, 1).value) for i in range(2, 102)]
        if len(set(a_ids)) != 100 or len(set(b_ids)) != 100:
            raise ValueError('제출 대상 ID 중복')
        print('기존 캐시 없이 공식 ZIP의 테스트 CSV에서 대상 항적을 읽습니다. 재학습은 하지 않습니다.', flush=True)
        rows = scan_zip(archive, set(a_ids) | set(b_ids), feature_names, args.chunk_size)
    ar = rows[rows.rfid_id.isin(a_ids)].reset_index(drop=True)
    br = rows[rows.rfid_id.isin(b_ids)].reset_index(drop=True)
    a, fallback = aggregate_a(ar, booster, feature_names, a_ids, args.threads)
    b, b_rows = baseline_b(br)
    path = args.output / 'submission.xlsx'
    save_workbook(template, a, b, path)
    a.to_csv(args.output / 'testA_predictions.csv', index=False, encoding='utf-8-sig')
    b_rows.to_csv(args.output / 'testB_anomaly_scores.csv', index=False, encoding='utf-8-sig')
    report = {'status': 'complete', 'file': str(path.resolve()), 'sha256': sha(path),
              'model': 'L127 + active row probability mean', 'testA_rows': len(ar), 'testB_rows': len(br),
              'testA_vessels': len(a_ids), 'testB_vessels': len(b_ids), 'active_fallback_vessels': fallback,
              'testB_incident_candidates': sum(item['incident_time'] is not None for item in b.values()),
              'testB_status': 'unvalidated temporary baseline; scores are not calibrated accident probabilities',
              'seconds': round(time.perf_counter() - start, 1)}
    (args.output / 'RUN_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
