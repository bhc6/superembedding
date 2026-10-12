"""Pinned CPU extraction of frozen pretrained OmniJev, with restartable shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MSO_FLA']='0';os.environ['MSO_BRANCH_CHUNK']='4'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.checkpoint import atomic_json,atomic_save,fingerprint
from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS
from superembedding.omnijev import load_omnijev
from scripts.train_general import pair,valid_records


def select_records(rows, per_source):
    sources=sorted({r['source'] for r in rows if r['split']=='train'})
    selected=[]
    for source in sources:
        train=sorted([r for r in rows if r['source']==source and r['split']=='train'],
                     key=lambda r:hashlib.sha256(('general-ablation-v1:'+r['id']).encode()).hexdigest())
        selected+=train[:per_source]
    selected += sorted([r for r in rows if r['split'] in {'val','ood'}],key=lambda r:r['id'])
    return selected


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--per-source',type=int,default=128);ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():raise RuntimeError('CPU extraction must not initialize CUDA')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw=(Path(args.data)/'records.jsonl').read_bytes()
    audit=json.loads((Path(args.data)/'audit.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=audit['manifest_sha256']:raise ValueError('Data audit mismatch')
    rows=list(valid_records([json.loads(s) for s in raw.decode('utf-8').splitlines()]))
    selected=select_records(rows,args.per_source)
    prov={'protocol':'general-frozen-head-ablation-v1','data_sha256':audit['manifest_sha256'],
        'ids':[r['id'] for r in selected],'train_per_source':args.per_source,'device':'cpu','dtype':'float32',
        'basis':list(GENERAL_BASIS.directions),'max_pixels':65536,'max_text_tokens':256,
        'supervision':'one fixed pair per image, random.Random(record id), no test selection',
        'adapter_sha256':hashlib.sha256(Path('models/OmniJev-0.8B-v1.1/adapter_model.safetensors').read_bytes()).hexdigest(),
        'base_sha256':hashlib.sha256(Path('models/Qwen3.5-0.8B/source_revision.json').read_bytes()).hexdigest(),
        'code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['scripts/extract_general_ablation.py','scripts/train_general.py','superembedding/general.py','superembedding/omnijev.py']},
        'torch':str(torch.__version__),'transformers':__import__('transformers').__version__}
    if (out/'provenance.json').exists() and json.loads((out/'provenance.json').read_text())!=prov:
        raise ValueError('Feature provenance mismatch')
    atomic_json(prov,out/'provenance.json');sig=fingerprint(prov)
    model=None;completed=0;started=time.perf_counter()
    for i,r in enumerate(selected):
        path=out/'shards'/f'{i:05d}.pt'
        if path.exists():
            saved=torch.load(path,weights_only=True)
            if saved['fingerprint']!=sig or saved['id']!=r['id']:raise ValueError('Feature cache mismatch')
            continue
        if shutil.disk_usage(out).free<700*2**20:raise RuntimeError('Insufficient disk reserve for feature extraction')
        if hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()!=r['image_file_sha256']:
            raise ValueError('Source image changed')
        if model is None:
            mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
            model=GeneralOmniEncoder(mso,GENERAL_BASIS,mode='raw',tune_lora=False).eval()
            if next(model.parameters()).device.type!='cpu':raise RuntimeError('Expected CPU model')
        tick=time.perf_counter();q,d,key=pair(r,random.Random(r['id']))
        uq,_=model.extract_features(q);ud,_=model.extract_features(d)
        if uq.shape!=(4,1024) or ud.shape!=(4,1024) or not torch.isfinite(uq).all() or not torch.isfinite(ud).all():
            raise ValueError('Unexpected frozen features')
        atomic_save({'id':r['id'],'source':r['source'],'split':r['split'],'key':key,
            'q':uq.float().cpu(),'d':ud.float().cpu(),'fingerprint':sig,
            'seconds':time.perf_counter()-tick},path)
        completed+=1
        status={'state':'extracting','pid':os.getpid(),'last_index':i,'total':len(selected),
            'completed_this_launch':completed,'seconds_this_launch':time.perf_counter()-started,
            'last_pair_seconds':time.perf_counter()-tick}
        atomic_json(status,out/'status.json')
        if completed%10==0 or completed==1:print(json.dumps(status),flush=True)
        if args.limit and completed>=args.limit:return
        if (out/'STOP_AFTER_SHARD').exists():
            atomic_json({**status,'state':'paused'},out/'status.json');return
    atomic_json({'state':'completed','pairs':len(selected),'fingerprint':sig},out/'status.json')
    print(json.dumps({'completed_pairs':len(selected)}),flush=True)


if __name__=='__main__':
    from filelock import FileLock
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',required=True)
    parsed,_=parser.parse_known_args();Path(parsed.output).mkdir(parents=True,exist_ok=True)
    with FileLock(str(Path(parsed.output)/'extract.lock'),timeout=0):main()
