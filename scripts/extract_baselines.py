"""Extract frozen mainstream features. Shards are atomic and restartable."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from PIL import Image
import torch
from superembedding.baselines import *
from superembedding.checkpoint import atomic_save,atomic_json,fingerprint

def api_key():
    import os
    # Never print configuration values, request bodies, headers, or HTTP exceptions.
    cfg={}
    if Path('.env').exists():
        for line in Path('.env').read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('\"').strip("'")
    key=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY') or cfg.get('QWEN_API_KEY') or cfg.get('DASHSCOPE_API_KEY')
    if not key: raise RuntimeError('Set DASHSCOPE_API_KEY or QWEN_API_KEY locally')
    return key

def api_batch(client,spec,root,rows,prompt):
    contents=[]
    for r in rows:
        data=image_bytes(root,r)
        with Image.open(io.BytesIO(data)) as im: mime=Image.MIME.get(im.format,'image/jpeg')
        contents.append({'image':f'data:{mime};base64,'+base64.b64encode(data).decode()})
    params=dict(spec['parameters'])
    if prompt: params['instruct']=prompt
    body={'model':spec['model'],'input':{'contents':contents},'parameters':params}
    start=time.perf_counter()
    for attempt in range(7):
        try:
            response=client.post(ENDPOINT,json=body)
        except (httpx.TimeoutException,httpx.TransportError):
            if attempt==6:raise RuntimeError('API network retries exhausted') from None
            time.sleep(min(2**attempt,30));continue
        if response.status_code==429 or response.status_code>=500:
            if attempt==6:raise RuntimeError(f'API retries exhausted: HTTP {response.status_code}')
            time.sleep(min(2**attempt,30));continue
        if response.status_code!=200:
            # Error strings can echo inputs; only emit HTTP and short provider error code.
            try: code=str(response.json().get('code','unknown'))[:80]
            except ValueError:code='non-json'
            raise RuntimeError(f'API HTTP {response.status_code}, code={code}')
        j=response.json();emb=j.get('output',{}).get('embeddings',[])
        if sorted(e['index'] for e in emb)!=list(range(len(rows))):
            raise ValueError('API returned incorrect item indices')
        v=torch.tensor([e['embedding'] for e in sorted(emb,key=lambda e:e['index'])],dtype=torch.float32)
        validate_vectors(v,len(rows),spec['width'])
        return v,{'usage':j.get('usage',{}),'request_id':j.get('request_id'),
            'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,'attempts':attempt+1,
            'request_fingerprint':fingerprint({'ids':[r['id'] for r in rows],'hashes':[r['image_sha256'] for r in rows],
                'model':spec['model'],'parameters':params})}

def local_model(backend,cache):
    from transformers import AutoImageProcessor,AutoModel,CLIPModel,SiglipModel
    from huggingface_hub import snapshot_download
    spec=MODELS[backend]
    folder=snapshot_download(spec['repo'],revision=spec['revision'],cache_dir=cache,
        allow_patterns=['config.json','preprocessor_config.json','model.safetensors','pytorch_model.bin']
        if backend=='clip_b32' else ['config.json','preprocessor_config.json','model.safetensors'])
    processor=AutoImageProcessor.from_pretrained(folder)
    cls=CLIPModel if backend=='clip_b32' else SiglipModel if backend=='siglip2_b16' else AutoModel
    model=cls.from_pretrained(folder).cuda().eval()
    return model,processor

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backends',nargs='+',choices=list(MODELS),required=True)
    ap.add_argument('--datasets',nargs='+',default=list(DATASETS))
    ap.add_argument('--root',default='runs/v4')
    ap.add_argument('--model-cache',default='models/v4')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit-batches',type=int,default=0)
    a=ap.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    from filelock import FileLock
    for backend in a.backends:
        spec=MODELS[backend];is_api='model' in spec; model=processor=None
        if is_api: client=httpx.Client(headers={'Authorization':'Bearer '+api_key()},timeout=120,limits=httpx.Limits(max_connections=a.workers))
        else: model,processor=local_model(backend,a.model_cache)
        for dataset in a.datasets:
            root,meta,records=read_data(dataset);out=Path(a.root)/dataset/backend
            out.parent.mkdir(parents=True,exist_ok=True)
            with FileLock(str(out)+'.lock'):
                prov=provenance(dataset,backend);initialize(out,prov)
                if (out/'features.pt').exists():
                    load_packed(out,records,prov);print(f'SKIP {dataset} {backend}',flush=True);continue
                batch=8 if is_api else 32
                groups=range(len(meta['directions'])) if spec.get('conditioned') else [0]
                jobs=[]
                for g in groups:
                    for offset in range(0,len(records),batch):
                        path=out/'shards'/f'{g}_{offset:06d}.pt';rows=records[offset:offset+batch]
                        if path.exists():
                            s=torch.load(path,weights_only=True)
                            if s['ids']!=[r['id'] for r in rows] or s['hashes']!=[r['image_sha256'] for r in rows]:raise ValueError('Resume shard mismatch')
                            validate_vectors(s['vectors'],len(rows),spec['width'])
                        else: jobs.append((g,offset,rows,path))
                start=time.perf_counter();completed=0
                if a.limit_batches: jobs=jobs[:a.limit_batches]
                def process(job):
                    g,offset,rows,path=job
                    prompt=instruction(meta['directions'][g]) if spec.get('conditioned') else None
                    if is_api:v,info=api_batch(client,spec,root,rows,prompt)
                    else:
                        images=[Image.open(io.BytesIO(image_bytes(root,r))).convert('RGB') for r in rows]
                        inputs=processor(images=images,return_tensors='pt').to('cuda')
                        with torch.no_grad():
                            if backend=='dinov2_small':v=model(**inputs).last_hidden_state[:,0]
                            else:
                                result=model.get_image_features(**inputs)
                                v=result.pooler_output if hasattr(result,'pooler_output') else result
                        v=v.float().cpu();validate_vectors(v,len(rows),spec['width']);info={}
                    atomic_save({'ids':[r['id'] for r in rows],'hashes':[r['image_sha256'] for r in rows],
                        'vectors':v,'info':info},path)
                    return len(rows)
                if is_api:
                    with ThreadPoolExecutor(a.workers) as pool:
                        for count in pool.map(process,jobs):
                            completed+=1
                            if completed%25==0:print(json.dumps({'backend':backend,'dataset':dataset,'batches':completed,'remaining_initial':len(jobs),'seconds':round(time.perf_counter()-start)}),flush=True)
                else:
                    for job in jobs:
                        process(job);completed+=1
                        if completed%25==0:print(json.dumps({'backend':backend,'dataset':dataset,'batches':completed,'remaining_initial':len(jobs),'seconds':round(time.perf_counter()-start)}),flush=True)
                if a.limit_batches:continue
                pack(out,records,spec,meta['directions'],batch)
                atomic_json({'count':len(records),'seconds_this_invocation':time.perf_counter()-start,
                    'batches_this_invocation':completed,'batch_size':batch,'backend':backend,'dataset':dataset},out/'extraction.json')
                print(f'COMPLETE {dataset} {backend}',flush=True)
        if is_api:client.close()
        else:del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
