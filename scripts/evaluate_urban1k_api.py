"""Resumable native Qwen API image/text retrieval on all Urban1k candidates."""
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
import torch
from torch.nn import functional as F
from scripts.extract_baselines import api_key
from superembedding.baselines import ENDPOINT
from superembedding.checkpoint import atomic_save,atomic_json,fingerprint

def main():
    root=Path('C:/Users/123/.cache/superembedding/v5/urban1k')
    rows=json.loads((root/'annotations/annotations.json').read_text(encoding='utf-8'))['annotations']
    out=Path('runs/v5/urban1k/qwen3_vl');out.mkdir(parents=True,exist_ok=True)
    if len(rows)!=1000:raise ValueError('Expected full Urban1k')
    with httpx.Client(timeout=120,headers={'Authorization':'Bearer '+api_key()}) as client:
        def run(job):
            kind,start=job;batch=rows[start:start+8];path=out/f'{kind}_{start:04d}.pt'
            ids=[r['image'] for r in batch]
            sig=fingerprint({'ids':ids,'captions':[r['caption'] for r in batch],'model':'qwen3-vl-embedding','dimension':2560})
            if path.exists():
                if torch.load(path,weights_only=True)['fingerprint']!=sig:raise RuntimeError('Cache mismatch')
                return
            valid=[i for i,r in enumerate(batch) if kind!='text' or r['caption'].strip()]
            contents=[{'text':batch[i]['caption']} if kind=='text' else {'image':'data:image/jpeg;base64,'+
                       base64.b64encode((root/'images'/batch[i]['image']).read_bytes()).decode()} for i in valid]
            body={'model':'qwen3-vl-embedding','input':{'contents':contents},'parameters':{'dimension':2560}}
            for attempt in range(7):
                try:r=client.post(ENDPOINT,json=body)
                except httpx.HTTPError:
                    if attempt==6:raise RuntimeError('API network retries exhausted') from None
                    time.sleep(min(2**attempt,30));continue
                if r.status_code in {429,500,502,503,504}:
                    time.sleep(min(2**attempt,30));continue
                if r.status_code!=200:raise RuntimeError(f'API HTTP {r.status_code}')
                j=r.json();emb=j['output']['embeddings']
                if sorted(e['index'] for e in emb)!=list(range(len(valid))):raise RuntimeError('API index mismatch')
                returned=torch.tensor([e['embedding'] for e in sorted(emb,key=lambda e:e['index'])])
                if returned.shape!=(len(valid),2560) or not torch.isfinite(returned).all():raise ValueError('Invalid API vector')
                vectors=torch.zeros(len(batch),2560);vectors[valid]=returned
                atomic_save({'vectors':F.normalize(vectors,dim=-1),'fingerprint':sig,
                    'valid_indices':valid,'empty_caption_policy':'zero placeholder, EXCLUDED from shared evaluation subset',
                    'request_id':j.get('request_id'),'usage':j.get('usage'),'utc':datetime.now(timezone.utc).isoformat()},path)
                return
            raise RuntimeError('API retries exhausted')
        with ThreadPoolExecutor(4) as pool:
            for i,_ in enumerate(pool.map(run,[(k,i) for k in ['text','image'] for i in range(0,1000,8)]),1):
                if i%25==0:print(json.dumps({'api_urban_batches':i,'total':250}),flush=True)
    def load(kind):return torch.cat([torch.load(out/f'{kind}_{i:04d}.pt',weights_only=True)['vectors'] for i in range(0,1000,8)])
    scores=load('image')@load('text').T;metrics={}
    kept=[i for i,r in enumerate(rows) if r['caption'].strip()]
    full_scores=scores;scores=scores[kept][:,kept];n=len(kept)
    for key,s in [('image_to_text',scores),('text_to_image',scores.T)]:
        ranked=s.argsort(descending=True,dim=-1)
        for k in [1,5,10]:metrics[f'{key}_R@{k}']=(ranked[:,:k]==torch.arange(n)[:,None]).any(-1).float().mean().item()
    result={'backend':'qwen3_vl','metrics':metrics,'queries':n,'candidates':n,
            'excluded_indices':[i for i in range(1000) if i not in kept],
            'scope':'Common nonempty-caption subset, NOT official full Urban1k. API rejects empty strings.',
            'dataset':json.loads((root/'source.json').read_text()),'dimension':2560,
            'conditioning':'native image/text, no additional instructions','training':False,
            'alias_warning':'Unversioned provider API; cannot identify with the open 2B/8B checkpoints',
            'annotation_sha256':hashlib.sha256((root/'annotations/annotations.json').read_bytes()).hexdigest()}
    atomic_save(full_scores,out/'scores.pt');atomic_json(result,'artifacts/results/v5/urban1k_qwen3_vl.json');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
