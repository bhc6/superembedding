"""Frozen Qwen modality interventions with identical cached target vectors."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from scripts.extract_baselines import api_key
from superembedding.baselines import ENDPOINT
from superembedding.checkpoint import atomic_save,atomic_json,fingerprint

def main():
    torch.set_num_threads(2)
    root=Path('C:/Users/123/.cache/superembedding/v5/mmeb_scienceqa')
    rows=pq.read_table(root/'test.parquet').to_pylist();texts=sorted({t for r in rows for t in r['tgt_text']})
    if len(rows)!=1000:raise RuntimeError('Require complete official task')
    index={t:i for i,t in enumerate(texts)}
    native=Path('runs/v5/scienceqa/qwen3_vl');out=Path('runs/v5/scienceqa_controls');out.mkdir(parents=True,exist_ok=True)
    targets=torch.cat([torch.load(native/f'target_{i:06d}.pt',weights_only=True)['vectors'] for i in range(0,len(texts),8)])
    original=torch.cat([torch.load(native/f'query_{i:06d}.pt',weights_only=True)['vectors'] for i in range(0,1000,8)])
    native_prov=json.loads((native/'provenance.json').read_text())
    hashes=[hashlib.sha256((root/r['qry_img_path']).read_bytes()).hexdigest() for r in rows]
    # Fixed cyclic shift, independent of predictions and labels; inspect equal image bytes.
    shift=137
    provenance={'native_provenance':native_prov,'shift':shift,
        'interventions':['text_only','image_only','shifted_image'],
        'target_vectors':'exact same frozen native target bank','target_sha256':hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
        'image_file_hashes_sha256':fingerprint(hashes),'seed':20261011,
        'same_image_bytes_after_shift':sum(hashes[i]==hashes[(i+shift)%1000] for i in range(1000))}
    path=out/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:raise RuntimeError('Control provenance changed')
    atomic_json(provenance,path)
    with httpx.Client(timeout=120,headers={'Authorization':'Bearer '+api_key()}) as client:
        def request(contents):
            body={'model':'qwen3-vl-embedding','input':{'contents':contents},'parameters':{'dimension':2560}}
            for attempt in range(7):
                try:r=client.post(ENDPOINT,json=body)
                except httpx.HTTPError:
                    if attempt==6:raise RuntimeError('API network retries exhausted') from None
                    time.sleep(min(2**attempt,30));continue
                if r.status_code in {429,500,502,503,504}:time.sleep(min(2**attempt,30));continue
                if r.status_code!=200:raise RuntimeError(f'API HTTP {r.status_code}')
                j=r.json();emb=j['output']['embeddings']
                if sorted(e['index'] for e in emb)!=list(range(len(contents))):raise RuntimeError('API index mismatch')
                v=torch.tensor([e['embedding'] for e in sorted(emb,key=lambda e:e['index'])])
                if v.shape!=(len(contents),2560) or not torch.isfinite(v).all() or (v.norm(dim=-1)==0).any():raise RuntimeError('Invalid API vector')
                return F.normalize(v,dim=-1),{'request_id':j.get('request_id'),'usage':j.get('usage'),
                    'utc':datetime.now(timezone.utc).isoformat()}
            raise RuntimeError('API server retries exhausted')
        def content(i,mode):
            obj={}
            if mode!='image_only':obj['text']=rows[i]['qry_text'].replace('<|image_1|>','')
            if mode!='text_only':
                image_idx=(i+shift)%1000 if mode=='shifted_image' else i
                obj['image']='data:image/jpeg;base64,'+base64.b64encode((root/rows[image_idx]['qry_img_path']).read_bytes()).decode()
            return obj
        drift_path=out/'native_drift_probe.pt'
        if not drift_path.exists():
            v,info=request([content(i,'full') for i in range(8)])
            atomic_save({'vectors':v,'info':info,'max_abs_error':float((v-original[:8]).abs().max()),
                         'min_cosine':float((v*original[:8]).sum(-1).min())},drift_path)
        drift=torch.load(drift_path,weights_only=True)
        if drift['min_cosine']<.9999:raise RuntimeError('API alias changed; re-establish contemporaneous full-input baseline')
        def run(job):
            mode,start=job;p=out/f'{mode}_{start:04d}.pt'
            sig=fingerprint({'mode':mode,'start':start,'provenance':provenance})
            if p.exists():
                if torch.load(p,weights_only=True)['fingerprint']!=sig:raise RuntimeError('Control cache mismatch')
                return
            v,info=request([content(i,mode) for i in range(start,min(start+8,1000))])
            atomic_save({'vectors':v,'info':info,'fingerprint':sig},p)
        jobs=[(mode,i) for mode in provenance['interventions'] for i in range(0,1000,8)]
        with ThreadPoolExecutor(4) as pool:
            for i,_ in enumerate(pool.map(run,jobs),1):
                if i%25==0:print(json.dumps({'control_batches':i,'total':len(jobs)}),flush=True)
    vectors={'full':original}
    for mode in provenance['interventions']:
        vectors[mode]=torch.cat([torch.load(out/f'{mode}_{i:04d}.pt',weights_only=True)['vectors'] for i in range(0,1000,8)])
    correctness={};predictions={}
    for mode,queries in vectors.items():
        correct=[];preds=[]
        for i,r in enumerate(rows):
            names=r['tgt_text'];order=sorted(range(len(names)),key=lambda k:hashlib.sha256(names[k].encode()).hexdigest())
            scores=queries[i]@targets[[index[t] for t in names]].T
            chosen=order[int(scores[order].argmax())];correct.append(names[chosen]==names[0]);preds.append(chosen)
        correctness[mode]=np.array(correct,dtype=np.float64);predictions[mode]=preds
    rng=np.random.default_rng(20261011);resamples=rng.integers(0,1000,size=(5000,1000));metrics={}
    for mode,c in correctness.items():
        delta=correctness['full']-c;ci=np.quantile(delta[resamples].mean(-1),[.025,.975]).tolist()
        metrics[mode]={'hit@1':float(c.mean()),'full_minus_control':float(delta.mean()),'paired_bootstrap_95':ci,
                       'full_correct_control_wrong':int(((correctness['full']==1)&(c==0)).sum()),
                       'full_wrong_control_correct':int(((correctness['full']==0)&(c==1)).sum())}
    report={'provenance':provenance,'metrics':metrics,'queries':1000,'predictions':predictions,
        'native_drift_probe':{k:v for k,v in drift.items() if k!='vectors'},
        'scope':'Task-specific modality intervention, not proof of reasoning or absence of pretraining leakage',
        'uncertainty':'5000 paired bootstrap samples over queries; shared answer/image dependence not clustered'}
    atomic_json(report,'artifacts/results/v5/scienceqa_modality_controls.json');print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
