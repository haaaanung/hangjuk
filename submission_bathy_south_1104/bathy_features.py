import json
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
STEEP=(0.5,2.0)
GDIR=Path(__file__).resolve().parent / "grid"
_G=dict(meta=json.loads((GDIR/"grid_meta.json").read_text(encoding="utf-8")),el=np.load(GDIR/"elevation.npy",mmap_mode="r"),tid=np.load(GDIR/"tid.npy",mmap_mode="r"),slope=np.load(GDIR/"slope_deg.npy",mmap_mode="r"))
def _row_stats(wkt, positions):
    m = _G["meta"]; el, tid, slope = _G["el"], _G["tid"], _G["slope"]
    geom = shapely.from_wkt(wkt)
    parts, prow = shapely.get_parts(geom, return_index=True)
    xy, ppart = shapely.get_coordinates(parts, return_index=True)
    row = prow[ppart]
    cs = m["cellsize"]; nr, nc = el.shape
    ci = np.floor((xy[:, 0] - m["xllcorner"]) / cs).astype(np.int64)
    ri = nr - 1 - np.floor((xy[:, 1] - m["yllcorner"]) / cs).astype(np.int64)
    ok = (ci >= 0) & (ci < nc) & (ri >= 0) & (ri < nr)
    ci, ri = np.clip(ci, 0, nc - 1), np.clip(ri, 0, nr - 1)
    e = el[ri, ci].astype("float64"); e[(~ok) | (e == -32768)] = np.nan
    d = -e; s = slope[ri, ci].astype("float64"); s[np.isnan(d)] = np.nan
    t = tid[ri, ci]
    df = pd.DataFrame({"row": row, "d": d, "s": s, "land": (e >= 0).astype(float), "st1": (s > STEEP[0]).astype(float),
                       "st2": (s > STEEP[1]).astype(float), "direct": ((t >= 10) & (t <= 17)).astype(float), "cell": ri * nc + ci})
    # along-track depth change between consecutive vertices of the same part only
    same = np.r_[False, ppart[1:] == ppart[:-1]]
    dd = np.abs(np.diff(d, prepend=np.nan)); dd[~same] = np.nan; df["dd"] = dd
    g = df.groupby("row")
    out = pd.DataFrame({"bathy_depth_mean": g.d.mean(), "bathy_depth_median": g.d.median(), "bathy_depth_min": g.d.min(),
                        "bathy_depth_max": g.d.max(), "bathy_depth_std": g.d.std(ddof=0), "bathy_slope_mean": g.s.mean(),
                        "bathy_slope_max": g.s.max(), "bathy_steep_frac_05": g.st1.mean(), "bathy_steep_frac_2": g.st2.mean(),
                        "bathy_land_frac": g.land.mean(), "bathy_ddepth_mean": g.dd.mean(), "bathy_direct_frac": g.direct.mean(),
                        "bathy_n_cells": g.cell.nunique(), "bathy_n_points": g.size()})
    out["bathy_depth_range"] = out.bathy_depth_max - out.bathy_depth_min
    out.index = positions[out.index.to_numpy()]
    return out
