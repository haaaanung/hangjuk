"""Train the frozen submission candidate from CV folds 1-5 only."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
def sha(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True); a=ap.parse_args()
    t=time.time(); p=a.project; cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    meta=json.loads((HERE/'model/model_l127.json').read_text(encoding='utf-8'))
    rows=pd.read_parquet(p/'outputs/cache/matched_vessel_separability_v1/rows.parquet', columns=['position','fold','label','vessel'])
    assert rows.fold.isin([1,2,3,4,5]).all() and rows.position.is_unique
    X=np.load(p/'outputs/cache/compliant_rowf1_v1/X_cur.npy',mmap_mode='r')
    assert X.shape==(len(rows),34)
    bathy=pd.read_parquet(p/'outputs/cache/bathymetry_v1/rows.parquet').loc[rows.position.to_numpy(),cfg['bathy_features']]
    assert np.array_equal(bathy.index,rows.position)
    features=meta['features']+cfg['bathy_features']; xa=np.c_[X,bathy.to_numpy('float32')]
    assert not np.isinf(xa).any()
    loc=[features.index(n) for n in ['center_lon','center_lat']]
    sig=[features.index(n) for n in ['num_points','points_per_segment','duplicate_ratio','num_segments','jitter_distance_fraction']]
    params=dict(meta['params']); params['interaction_constraints']=[[i for i in range(48) if i not in sig],[i for i in range(48) if i not in loc]]
    print(f'TRAIN rows={len(rows):,} vessels={rows.vessel.nunique()} features={len(features)} folds=1..5',flush=True)
    mdl=lgb.LGBMClassifier(**params).fit(xa,rows.label.to_numpy())
    assert mdl.classes_.tolist()==[0,1,2,3]
    target=HERE/'model/model_bathy_ic.txt'; mdl.booster_.save_model(str(target))
    info=dict(params=params,features=features,classes=meta['classes'],train_rows=len(rows),train_vessels=int(rows.vessel.nunique()),train_folds=[1,2,3,4,5],model_sha256=sha(target),seconds=time.time()-t)
    (HERE/'model/model_bathy_ic.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
