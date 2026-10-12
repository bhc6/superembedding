"""Full-task frozen evaluations; native API and explicit dual-encoder controls."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from PIL import Image
from superembedding.baselines import MODELS,ENDPOINT
from superembedding.checkpoint import atomic_json,atomic_save,fingerprint
from scripts.extract_baselines import api_key,local_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['qwen3_vl','clip_b32','siglip2_b16','omnijev_raw','general'],required=True)
    ap.add_argument('--checkpoint');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--query-mode',choices=['full','text_only'],default='full')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--data',default='C:/Users/123/.cache/superembedding/v5/mmeb_scienceqa')
    args=ap.parse_args();torch.set_num_threads(2 if args.device=='cpu' else 4)
    if args.device=='cpu':
        if args.backend in {'clip_b32','siglip2_b16'}:raise ValueError('Dual-encoder CPU mode not implemented')
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
    root=Path(args.data);rows=pq.read_table(root/'test.parquet').to_pylist()
    if len(rows)!=1000:raise ValueError('Expected all 1000 official queries')
    if not (root/'metadata.json').exists():raise RuntimeError('Dataset download not complete')
    texts=sorted({text for row in rows for text in row['tgt_text']})
    index={s:i for i,s in enumerate(texts)}
    spec=MODELS.get(args.backend,{})
    prov={'backend':args.backend,'spec':spec,'dataset':json.loads((root/'metadata.json').read_text()),
          'query':'original qry_text with image placeholder removed; no additional task instruction',
          'candidate':'raw original target text; no task instruction',
          'metric':'hit@1 on each original candidate list; target text equals first entry',
          'api_warning':'API alias does not identify local Qwen2B/8B weights' if args.backend=='qwen3_vl' else None}
    if args.checkpoint:prov['checkpoint_sha256']=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    tag=args.backend+('_'+prov['checkpoint_sha256'][:12] if args.checkpoint else '')
    if args.device=='cpu':
        prov['execution']='cpu float32';tag+='_cpu'
    native_out=Path('runs/v5/scienceqa')/tag
    if args.query_mode=='text_only':
        native_prov=json.loads((native_out/'provenance.json').read_text(encoding='utf-8'))
        if native_prov!=prov:raise ValueError('Full-input run provenance mismatch')
        if not (Path('artifacts/results/v5')/f'scienceqa_{tag}.json').exists():
            raise ValueError('Complete full-input evaluation before text-only control')
        prov={**prov,'query_mode':'text_only','control':'Remove query images only; reuse exact full-input target vectors'}
        tag+='_text_only'
    out=Path('runs/v5/scienceqa')/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'provenance.json').exists() and json.loads((out/'provenance.json').read_text())!=prov:
        raise RuntimeError('Provenance changed')
    atomic_json(prov,out/'provenance.json')
    if args.backend=='qwen3_vl':
        client=httpx.Client(headers={'Authorization':'Bearer '+api_key()},timeout=120)
        def encode(objects):
            contents=[]
            for obj in objects:
                content={}
                if obj.get('text'):content['text']=obj['text']
                if obj.get('image'):
                    content['image']='data:image/jpeg;base64,'+base64.b64encode(Path(obj['image']).read_bytes()).decode()
                contents.append(content)
            body={'model':spec['model'],'input':{'contents':contents},'parameters':spec['parameters']}
            for attempt in range(7):
                try:r=client.post(ENDPOINT,json=body)
                except httpx.HTTPError:
                    if attempt==6:raise RuntimeError('API transport retries exhausted') from None
                    time.sleep(min(2**attempt,30));continue
                if r.status_code in {429,500,502,503,504}:
                    time.sleep(min(2**attempt,30));continue
                if r.status_code!=200:raise RuntimeError(f'API error HTTP {r.status_code}')
                j=r.json();emb=j.get('output',{}).get('embeddings',[])
                if sorted(e['index'] for e in emb)!=list(range(len(objects))):raise ValueError('API index mismatch')
                vectors=torch.tensor([e['embedding'] for e in sorted(emb,key=lambda e:e['index'])])
                return vectors,{'request_id':j.get('request_id'),'usage':j.get('usage'),
                                'utc':datetime.now(timezone.utc).isoformat()}
            raise RuntimeError('API rate/server retries exhausted')
    elif args.backend in {'clip_b32','siglip2_b16'}:
        model,processor=local_model(args.backend,'C:/Users/123/.cache/superembedding/v4/models')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['revision'])
        def encode(objects):
            # Standard equal-weight fusion is an explicit heuristic baseline.
            # These models have no native cross-attention multimodal query encoder.
            with torch.no_grad():
                max_len=77 if args.backend=='clip_b32' else 64
                tokens=tokenizer([o.get('text','') for o in objects],padding='max_length',truncation=True,max_length=max_len,return_tensors='pt').to('cuda')
                result=model.get_text_features(**tokens)
                vectors=result.pooler_output if hasattr(result,'pooler_output') else result
                vectors=F.normalize(vectors.float(),dim=-1)
                if objects[0].get('image'):
                    images=[Image.open(o['image']).convert('RGB') for o in objects]
                    result=model.get_image_features(**processor(images=images,return_tensors='pt').to('cuda'))
                    image_vectors=result.pooler_output if hasattr(result,'pooler_output') else result
                    vectors=F.normalize(vectors+F.normalize(image_vectors.float(),dim=-1),dim=-1)
                return vectors.cpu(),{'fusion':'equal normalized image+text; native text targets'}
    else:
        os.environ['MSO_FLA']='0';os.environ['MSO_BRANCH_CHUNK']='4'
        from superembedding.omnijev import load_omnijev
        from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS
        if args.backend=='general':
            if not args.checkpoint:raise ValueError('General backend requires a checkpoint')
            from superembedding.general_export import load_bundle_encoder
            model,saved=load_bundle_encoder(args.checkpoint)
        else:
            mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
            model=GeneralOmniEncoder(mso,GENERAL_BASIS,mode='raw',tune_lora=False).eval()
        def encode(objects):
            vectors=[]
            with torch.no_grad():
                for obj in objects:
                    state={'text':obj.get('text',''),'images':[obj['image']] if obj.get('image') else []}
                    if args.backend=='omnijev_raw':
                        u,_=model.extract_features(state);v=F.normalize(F.normalize(u.float(),dim=-1).flatten(),dim=-1)
                    else:v=model(state).embedding
                    vectors.append(v.cpu())
            return torch.stack(vectors),{'basis':'fixed general evidence directions','max_pixels':65536}
    batch_size=8
    jobs=[]
    queries=[{'text':r['qry_text'].replace('<|image_1|>',''),
              **({'image':str(root/r['qry_img_path'])} if args.query_mode=='full' else {})} for r in rows]
    for kind,objects in [('target',[{'text':t} for t in texts]),('query',queries)]:
        for offset in range(0,len(objects),batch_size):
            batch=objects[offset:offset+batch_size];path=out/f'{kind}_{offset:06d}.pt'
            sig=fingerprint({'objects':batch,'provenance':prov})
            if kind=='target' and args.query_mode=='text_only':
                saved=torch.load(native_out/f'{kind}_{offset:06d}.pt',weights_only=True)
                expected=fingerprint({'objects':batch,'provenance':native_prov})
                if saved['fingerprint']!=expected:raise ValueError('Full-input target cache mismatch')
                if path.exists():
                    current=torch.load(path,weights_only=True)
                    if current['fingerprint']!=sig or not torch.equal(current['vectors'],saved['vectors']):
                        raise ValueError('Control target bank differs')
                else:atomic_save({**saved,'fingerprint':sig,'full_input_target_fingerprint':expected},path)
            elif path.exists():
                saved=torch.load(path,weights_only=True)
                if saved['fingerprint']!=sig:raise RuntimeError('Cached request changed')
            else:jobs.append((batch,path,sig))
    def run(job):
        batch,path,sig=job;v,info=encode(batch)
        if v.ndim!=2 or len(v)!=len(batch) or not torch.isfinite(v).all() or (v.norm(dim=-1)==0).any():raise ValueError('Invalid embeddings')
        atomic_save({'vectors':F.normalize(v.float(),dim=-1),'fingerprint':sig,'info':info},path)
    if args.backend=='qwen3_vl':
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,_ in enumerate(pool.map(run,jobs),1):
                if i%25==0:print(json.dumps({'backend':tag,'batches':i,'total':len(jobs)}),flush=True)
        client.close()
    else:
        for i,job in enumerate(jobs,1):
            run(job)
            if i%25==0:print(json.dumps({'backend':tag,'batches':i,'total':len(jobs)}),flush=True)
    def load(kind,n):return torch.cat([torch.load(out/f'{kind}_{i:06d}.pt',weights_only=True)['vectors'] for i in range(0,n,batch_size)])
    q=load('query',len(rows));d=load('target',len(texts));correct=[];predictions=[]
    for i,row in enumerate(rows):
        candidates=row['tgt_text'];scores=q[i]@d[[index[s] for s in candidates]].T
        # Deterministic text-based tie order avoids positive-at-position-zero advantage.
        order=sorted(range(len(candidates)),key=lambda k:hashlib.sha256(candidates[k].encode()).hexdigest())
        selected=order[int(scores[order].argmax())]
        correct.append(candidates[selected]==candidates[0]);predictions.append({'query':i,'candidate_index':selected,'correct':correct[-1]})
    n=len(correct);p=sum(correct)/n;z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n);half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    result={'backend':tag,'queries':n,'unique_targets':len(texts),'hit@1':p,'wilson_95':[center-half,center+half],
            'candidate_count_min':min(len(r['tgt_text']) for r in rows),'candidate_count_max':max(len(r['tgt_text']) for r in rows),
            'provenance':prov,'local_evaluation':True,'training_on_this_task':False,
            'scope':'Full ScienceQA task from pinned MMEB-v1 metadata; NOT full MMEB or proof of SOTA.',
            'predictions':predictions}
    from superembedding.validation_protocol import question_type
    groups={}
    for i,row in enumerate(rows):
        typ=question_type(row['qry_text'].replace('<|image_1|>',''),'scienceqa')
        groups.setdefault(typ,[]).append(bool(correct[i]))
    result['question_types']={typ:{'queries':len(hits),'hit@1':sum(hits)/len(hits)} for typ,hits in groups.items()}
    result['question_type_annotation']='Question-only heuristic, not official ScienceQA question types'
    if args.query_mode=='text_only':
        import numpy as np
        full=json.loads((Path('artifacts/results/v5')/f"scienceqa_{tag.removesuffix('_text_only')}.json").read_text(encoding='utf-8'))
        if [p['query'] for p in full['predictions']]!=list(range(n)):raise ValueError('Full-input query alignment mismatch')
        delta=np.array([float(p['correct'])-float(c) for p,c in zip(full['predictions'],correct)])
        rng=np.random.default_rng(20261011)
        ci=np.quantile(delta[rng.integers(0,n,size=(5000,n))].mean(-1),[.025,.975]).tolist()
        result['full_minus_text_only']={'delta':float(delta.mean()),'paired_bootstrap_95':ci,
            'unit':'query; shared answer/image dependence not clustered',
            'full_result_sha256':hashlib.sha256((Path('artifacts/results/v5')/f"scienceqa_{tag.removesuffix('_text_only')}.json").read_bytes()).hexdigest()}
    atomic_json(result,Path('artifacts/results/v5')/f'scienceqa_{tag}.json')
    print(json.dumps({k:v for k,v in result.items() if k not in {'predictions','provenance'}}),flush=True)

if __name__=='__main__':main()
