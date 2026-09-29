"""STEP 1~2: 원본 전체의 표 구조를 확인하고 실제 선분을 관찰한다."""
from collections import Counter
import contextlib
import json
import pandas as pd
import shapely
from src.common import ROOT, OUTPUT, FIGURES, data_files, draw_track, plt, clean_metadata


def main():
    tables, summaries, rejected = [], [], []
    with (OUTPUT / 'data_check.txt').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        for path in data_files():
            missing, dtypes, count = Counter(), {}, 0
            for chunk in pd.read_csv(path, chunksize=20000, low_memory=False):
                if count == 0:
                    print('\nFILE', path.name)
                    print('df.head()\n', chunk.head().to_string(max_colwidth=100))
                    print('df.columns', chunk.columns.tolist())
                count += len(chunk)
                missing.update(chunk.isna().sum().to_dict())
                for name in chunk:
                    dtypes.setdefault(name, set()).add(str(chunk[name].dtype))
                meta, valid = clean_metadata(chunk)
                bad = chunk.loc[~valid].copy()
                if len(bad):
                    bad.insert(0, 'source_row', bad.index)
                    bad.insert(0, 'source_file', path.name)
                    rejected.append(bad)
                tables.append(meta.loc[valid])
            print('df.shape', (count, len(missing)))
            print('df.dtypes', {k: sorted(v) for k, v in dtypes.items()})
            print('df.isna().sum()', dict(missing))
            summaries.append({'file': path.name, 'rows': count, 'columns': len(missing), 'missing': dict(missing)})
        meta = pd.concat(tables, ignore_index=True)
        summary = {
            'files': summaries, 'raw_rows': sum(s['rows'] for s in summaries),
            'valid_metadata_rows': len(meta), 'vessels': meta.rfid_id.nunique(),
            'repeated_vessels': int((meta.rfid_id.value_counts() > 1).sum()),
            'date_min': int(meta.part_dt.min()), 'date_max': int(meta.part_dt.max()),
            'dates': meta.part_dt.nunique(),
            'time_values': sorted(meta.rfid_time.unique().tolist()),
            'label_counts': meta.fishing_code.value_counts().to_dict(),
            'vessels_per_label': meta.groupby('fishing_code').rfid_id.nunique().to_dict(),
            'labels_per_vessel': meta.groupby('rfid_id').fishing_code.nunique().value_counts().to_dict(),
            'duplicate_vessel_date_time': int(meta.duplicated(['rfid_id', 'part_dt', 'rfid_time']).sum()),
            'duplicate_vessel_date': int(meta.duplicated(['rfid_id', 'part_dt']).sum()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if rejected:
            pd.concat(rejected).to_csv(OUTPUT / 'invalid_source_rows.csv', index=False, encoding='utf-8-sig')
        xlsx = pd.read_excel(next(ROOT.glob('*.xlsx')))
        print('ACCIDENT', xlsx.shape, xlsx.columns.tolist())
        print(xlsx.dtypes, xlsx.isna().sum())
        sample = pd.read_csv(data_files()[0], nrows=2000)
        examples = sample.iloc[[0, 2, 3]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (idx, row) in zip(axes, examples.iterrows()):
            lines = list(shapely.from_wkt(row.geom).geoms)
            print('EXAMPLE', idx, 'segments', len(lines), 'points', sum(len(s.coords) for s in lines))
            for j, line in enumerate(lines):
                print('segment', j, 'points', len(line.coords), 'start', line.coords[0], 'end', line.coords[-1])
            draw_track(ax, row.geom)
            ax.set_title(f'원본 행 {idx}: {row.fishing_code}\n{len(lines)} segments')
        fig.tight_layout(); fig.savefig(FIGURES / 'raw_tracks.png', dpi=160); plt.close(fig)
    (OUTPUT / 'data_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('상세 실제 출력: outputs/data_check.txt')


if __name__ == '__main__':
    main()
