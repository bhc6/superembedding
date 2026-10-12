"""Qwen native cached vectors on original paired evidence diagnostic."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
import torch
from torch.nn import functional as F
from scripts.extract_baselines import api_key
from superembedding.baselines import ENDPOINT
from superembedding.checkpoint import atomic_save,atomic_json,fingerprint

def main():
    root=Path('data/evidence_diagnostic_v1');out=Path('runs/v5/evidence/qwen3_vl');out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    meta=json.loads((root/'metadata.json').read_text())
    image_names=sorted({r[k] for r in records for k in ['anchor','relevant_change','irrelevant_change']})
    texts=sorted({r['query'] for r in records});objects=[{'image':p} for p in image_names]+[{'text':t} for t in texts]
    with httpx.Client(timeout=120,headers={'Authorization':'Bearer '+api_key()}) as client:
        def run(start):
            batch=objects[start:start+8];path=out/f'{start:04d}.pt'
            sig=fingerprint({'objects':batch,'dataset':meta['records_sha256'],'model':'qwen3-vl-embedding','dimension':2560})
            if path.exists():
                if torch.load(path,weights_only=True)['fingerprint']!=sig:raise RuntimeError('Diagnostic cache mismatch')
                return
            contents=[{'image':'data:image/png;base64,'+base64.b64encode((root/o['image']).read_bytes()).decode()} if 'image' in o else o for o in batch]
            for attempt in range(7):
                try:r=client.post(ENDPOINT,json={'model':'qwen3-vl-embedding','input':{'contents':contents},'parameters':{'dimension':2560}})
                except httpx.HTTPError:
                    if attempt==6:raise RuntimeError('API network failure') from None
                    time.sleep(min(2**attempt,30));continue
                if r.status_code in {429,500,502,503,504}:time.sleep(min(2**attempt,30));continue
                if r.status_code!=200:raise RuntimeError(f'API HTTP {r.status_code}')
                j=r.json();emb=j['output']['embeddings']
                if sorted(e['index'] for e in emb)!=list(range(len(batch))):raise ValueError('API index mismatch')
                v=torch.tensor([e['embedding'] for e in sorted(emb,key=lambda e:e['index'])])
                if v.shape!=(len(batch),2560) or not torch.isfinite(v).all():raise ValueError('Invalid API vectors')
                atomic_save({'vectors':F.normalize(v,dim=-1),'fingerprint':sig,'request_id':j.get('request_id'),
                             'usage':j.get('usage'),'utc':datetime.now(timezone.utc).isoformat()},path);return
            raise RuntimeError('API retries exhausted')
        with ThreadPoolExecutor(4) as pool:
            for i,_ in enumerate(pool.map(run,range(0,len(objects),8)),1):
                if i%20==0:print(json.dumps({'diagnostic_batches':i}),flush=True)
    v=torch.cat([torch.load(out/f'{start:04d}.pt',weights_only=True)['vectors'] for start in range(0,len(objects),8)])
    images={name:v[i] for i,name in enumerate(image_names)};queries={text:v[len(image_names)+i] for i,text in enumerate(texts)}
    cases=[]
    for r in records:
        a,b,c=[images[r[k]] for k in ['anchor','relevant_change','irrelevant_change']];q=queries[r['query']]
        sa,sb,sc=[float(q@x) for x in [a,b,c]]
        cases.append({'id':r['id'],'scene_group':r['scene_group'],'attribute':r['attribute'],
                      'anchor_margin':sa-sb,'nuisance_margin':sc-sb,'nuisance_query_shift':abs(sa-sc),
                      'target_change_distance':float(1-a@b),'nuisance_distance':float(1-a@c)})
    metrics={}
    for attribute in sorted({r['attribute'] for r in cases}):
        group=[r for r in cases if r['attribute']==attribute];n=len(group)
        metrics[attribute]={'cases':n,'anchor_preference':sum(r['anchor_margin']>0 for r in group)/n,
                            'robust_both_preference':sum(min(r['anchor_margin'],r['nuisance_margin'])>0 for r in group)/n,
                            'mean_target_distance':sum(r['target_change_distance'] for r in group)/n,
                            'mean_nuisance_distance':sum(r['nuisance_distance'] for r in group)/n,
                            'mean_nuisance_query_shift':sum(r['nuisance_query_shift'] for r in group)/n}
    result={'backend':'qwen3-vl-embedding API 2560 native','dataset_sha256':meta['records_sha256'],
            'metrics':metrics,'cases':cases,'scope':'Synthetic diagnostic, no training, no natural-world or SOTA claim',
            'candidate_encoding':'Each image encoded once independently of query; reused across directions',
            'case_dependence':'Four interventions share each base scene; uncertainty must cluster by scene.'}
    atomic_json(result,'artifacts/results/v5/evidence_qwen3_vl.json');print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
