"""Generate one submission workbook; preserve the baseline Test B exactly."""
import argparse, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from openpyxl import load_workbook
import base_predict as base

HERE=Path(__file__).resolve().parent
def group_mean(p,ids,targets):
    u,inv=np.unique(ids,return_inverse=True); n=np.bincount(inv)
    v=np.stack([np.bincount(inv,weights=p[:,j]) for j in range(4)],axis=1)/n[:,None]
    return pd.DataFrame(v,index=u).loc[targets].to_numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',type=Path,required=True,dest='zip_path'); ap.add_argument('--chunk-size',type=int,default=10000); a=ap.parse_args()
    t=time.time(); cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    bm=json.loads((HERE/'model/model_l127.json').read_text(encoding='utf-8')); em=json.loads((HERE/'model/model_bathy_ic.json').read_text(encoding='utf-8'))
    for name,meta in [('model_l127.txt',bm),('model_bathy_ic.txt',em)]:
        assert base.sha(HERE/'model'/name)==meta['model_sha256']
    baseline=load_workbook(io.BytesIO((HERE/'baseline_workbook.bin').read_bytes())); sa=baseline['테스트A']; ids=[int(sa.cell(i,1).value) for i in range(2,sa.max_row+1)]
    assert len(ids)==len(set(ids))==100
    old={int(sa.cell(i,1).value):sa.cell(i,2).value for i in range(2,sa.max_row+1)}
    with zipfile.ZipFile(a.zip_path) as z:
        samples=[e for e in z.infolist() if base.decoded_name(e.filename).endswith('/제출샘플.xlsx')]; assert len(samples)==1
        template=z.read(samples[0]); assert hashlib.sha256(template).hexdigest()=='035fbced3dd46319f2506ab9b8bbb4f1756712ae3d94ecf6953d3587e056609c'
        wb=load_workbook(io.BytesIO(template),data_only=True)
        assert [int(wb['테스트A'].cell(i,1).value) for i in range(2,102)]==ids
        rows=base.scan_zip(z,set(ids),em['features'],a.chunk_size)
    mb=lgb.Booster(model_str=(HERE/'model/model_l127.txt').read_text(encoding='utf-8')); me=lgb.Booster(model_str=(HERE/'model/model_bathy_ic.txt').read_text(encoding='utf-8'))
    pb=mb.predict(rows[bm['features']].to_numpy('float32'),num_threads=8)
    south=(rows.center_lon>=126.5)&(rows.center_lon<129.3)&(rows.center_lat<35.3)
    pred=pb.copy()
    if south.any(): pred[south]=me.predict(rows.loc[south,em['features']].to_numpy('float32'),num_threads=8)
    assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
    np.testing.assert_allclose(pred.sum(1),1,atol=1e-6)
    np.testing.assert_array_equal(pred[~south],pb[~south])
    avg=group_mean(pred,rows.rfid_id.to_numpy(),ids); bavg=group_mean(pb,rows.rfid_id.to_numpy(),ids)
    labels=np.array(base.CLASSES)[avg.argmax(1)]; conf=avg.max(1)
    before_b=list(baseline['테스트B'].values)
    headers={c.value:c.column for c in sa[1]}
    for i,(label,prob) in enumerate(zip(labels,conf),2):
        sa.cell(i,headers['예측 조업형태'],str(label)); sa.cell(i,headers['예측조업형태 확률'],float(prob))
    target=HERE/'submission.xlsx'; baseline.save(target)
    check=load_workbook(target,data_only=True)
    assert list(check['테스트B'].values)==before_b
    assert [int(check['테스트A'].cell(i,1).value) for i in range(2,102)]==ids
    assert all(check['테스트A'].cell(i,2).value in base.CLASSES and 0<=check['테스트A'].cell(i,3).value<=1 for i in range(2,102))
    changes=[{'rfid_id':rid,'old':old[rid],'new':str(label),'probability':float(prob),'baseline_allmean':base.CLASSES[int(bavg[j].argmax())]} for j,(rid,label,prob) in enumerate(zip(ids,labels,conf))]
    (HERE/'comparison.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'status':'complete','model':'Bathy14 + IC in South, L127 elsewhere; all-row mean','testA_rows':len(rows),'testA_vessels':len(ids),'south_rows':int(south.sum()),'south_vessels':int(rows.loc[south,'rfid_id'].nunique()),'changed_vs_existing_submission':sum(old[rid]!=label for rid,label in zip(ids,labels)),'changed_vs_L127_allmean':int((avg.argmax(1)!=bavg.argmax(1)).sum()),'class_counts':dict(zip(*[x.tolist() for x in np.unique(labels,return_counts=True)])),'testB_cell_values_unchanged':True,'submission_sha256':base.sha(target),'seconds':round(time.time()-t,1),'test_score':None,'aggregation_note':'Existing submission uses active-row mean; new candidate uses the all-row mean fixed during validation.'}
    (HERE/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
