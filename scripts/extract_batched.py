"""Batched fixed-direction features with per-image restart and numerical audit."""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from torch.nn import functional as F
from filelock import FileLock
from superembedding import AttributeBasis
from superembedding.checkpoint import atomic_save,atomic_json
from superembedding.omnijev import load_omnijev,OmniJevEmbeddingEncoder
from superembedding.batched import prepare_image,encode_prepared

def extract(args):
    root=Path(args.data).resolve(); out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'metadata.json').read_text()); basis=AttributeBasis(tuple(meta['directions']),meta['instruction'])
    prov={'adapter':str(Path('models/OmniJev-0.8B-v1.1').resolve()),'base':str(Path('models/Qwen3.5-0.8B').resolve()),
        'basis':{'directions':list(basis.directions),'instruction':basis.instruction},'max_pixels':65536,
        'dataset_revision':meta['revision'],'encoding':'same_layout_image_batch_v1','batch_size':args.batch_size}
    if args.disable_lora: prov['disable_lora']=True
    manifest=out/'provenance.json'
    if manifest.exists() and json.loads(manifest.read_text())!=prov: raise ValueError('Cache settings changed')
    atomic_json(prov,manifest)
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    todo=[r for r in records if not (out/f"{r['id']}.pt").exists()]
    if not todo: print('All batched features already cached.',flush=True); return
    torch.set_num_threads(4); start=time.perf_counter()
    mso=load_omnijev(prov['adapter'],prov['base'],max_pixels=65536)
    encoder=OmniJevEmbeddingEncoder(mso,basis).eval();torch.cuda.reset_peak_memory_stats()
    pending=OrderedDict(); count=0; audit=[]
    def flush(key):
        nonlocal count
        group=pending.pop(key); prepared=[p for r,p in group]
        with encoder.backbone.disable_adapter() if args.disable_lora else nullcontext():
            u,zq=encode_prepared(encoder,prepared)
            # Only training examples are used for the numerical implementation audit.
            for j,(r,p) in enumerate(group):
                if len(audit)>=16 or r['split']!='train': continue
                refu,refq=encoder.extract_features({'images':[str(root/r['image'])]})
                cossim=min(F.cosine_similarity(u[j],refu,dim=-1).min().item(),F.cosine_similarity(zq[j],refq,dim=-1).item())
                item={'id':r['id'],'batch':len(group),'minimum_cosine_to_official':cossim,
                      'u_max_abs_diff':(u[j]-refu).abs().max().item(),'zq_max_abs_diff':(zq[j]-refq).abs().max().item()}
                if cossim<.999: raise RuntimeError(f'Batched branch equivalence failed: {item}')
                audit.append(item)
        for j,(r,_) in enumerate(group):
            atomic_save({'u':u[j].cpu(),'zq':zq[j].cpu(),'id':r['id'],'image_sha256':r['image_sha256']},out/f"{r['id']}.pt")
        count+=len(group)
        if count%100<len(group) or count==len(todo):
            print(json.dumps({'done':count,'total':len(todo),'seconds_per_image_including_setup':(time.perf_counter()-start)/count,
                              'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20}),flush=True)
    for r in todo:
        p=prepare_image(encoder,root/r['image'])
        p['prefix']={k:v.cpu() for k,v in p['prefix'].items()}
        key=p['key'];pending.setdefault(key,[]).append((r,p))
        if len(pending[key])>=args.batch_size: flush(key)
        elif sum(map(len,pending.values()))>=64: flush(next(iter(pending)))
    for key in list(pending): flush(key)
    result={'count_this_invocation':count,'wall_seconds':time.perf_counter()-start,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'batch_size':args.batch_size,
        'numerical_audit':audit,'cosine_tolerance':.999,'bitwise_equal_to_serial':False}
    atomic_json(result,out/'extraction.json');print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--disable-lora',action='store_true')
    args=ap.parse_args();Path(args.output).resolve().parent.mkdir(parents=True,exist_ok=True)
    if args.batch_size<1: raise ValueError('Batch size must be positive')
    with FileLock(str(Path(args.output).resolve())+'.lock'): extract(args)
