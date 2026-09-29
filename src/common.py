from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs'
FIGURES = OUTPUT / 'figures'
METRICS = OUTPUT / 'metrics'
for folder in [OUTPUT, FIGURES, METRICS]:
    folder.mkdir(parents=True, exist_ok=True)
if 'Malgun Gothic' in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def data_files():
    return sorted(ROOT.glob('vpass*/*/*.csv'))


def clean_metadata(chunk):
    import pandas as pd
    keys = ['part_dt', 'rfid_id', 'rfid_time', 'fishing_code']
    meta = chunk[keys].copy()
    for col in keys[:3]:
        meta[col] = pd.to_numeric(meta[col], errors='coerce')
    dates = pd.to_datetime(meta.part_dt.astype('string').str.replace(r'\.0$', '', regex=True),
                           format='%Y%m%d', errors='coerce')
    valid = dates.between('2026-01-01', '2026-06-30')
    valid &= meta.rfid_id.notna() & (meta.rfid_id > 0) & (meta.rfid_id % 1 == 0)
    valid &= meta.rfid_time.between(0, 23) & (meta.rfid_time % 1 == 0)
    valid &= meta.fishing_code.notna()
    extra = chunk.filter(like='Unnamed:')
    valid &= ~extra.notna().any(axis=1)
    return meta, valid


def draw_track(ax, wkt):
    import shapely
    import numpy as np
    g = shapely.from_wkt(wkt)
    for line in shapely.get_parts(g):
        xy = np.asarray(line.coords)
        ax.plot(xy[:, 0], xy[:, 1], '.-', linewidth=.8, markersize=2)
        ax.plot(xy[0, 0], xy[0, 1], 'go', markersize=3)
        ax.plot(xy[-1, 0], xy[-1, 1], 'rx', markersize=4)
    ax.set_aspect(1 / np.cos(np.radians(g.centroid.y)), adjustable='datalim')
    if g.bounds[0] == g.bounds[2] and g.bounds[1] == g.bounds[3]:
        ax.set_xlim(g.centroid.x - .0001, g.centroid.x + .0001)
        ax.set_ylim(g.centroid.y - .0001, g.centroid.y + .0001)
    ax.set_xlabel('경도'); ax.set_ylabel('위도')
    ax.ticklabel_format(useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=.2)
