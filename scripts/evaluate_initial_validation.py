"""One backbone pass yields raw and reconstructed Step-0 CUDA/CPU controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--audit',required=True)
    ap.add_argument('--data',default='D:/DEMO/superembedding-storage/superembedding/v5/data_v51')
    ap.add_argument('--cache',default='D:/DEMO/superembedding-storage/superembedding/v5/initial_validation')
    args=ap.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['MSO_FLA']='0';os.environ['MSO_BRANCH_CHUNK']='4';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch.nn import functional as F
    from filelock import FileLock
    from superembedding.checkpoint import atomic_json,atomic_save,fingerprint
    from superembedding.general_export import load_bundle_encoder
    from superembedding.validation_protocol import load_protocol,score_vectors,paired_delta
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol,items,_=load_protocol(args.data)
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if not audit['cpu_rng_matches_parent'] or audit['device']!=args.device:raise ValueError('Invalid reconstruction audit')
    prov={'protocol':protocol,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
          'reconstruction':audit,'device':args.device,'dtype':'bfloat16' if args.device=='cuda' else 'float32',
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'Same queries, candidates and precision as training when CUDA; Step 0 is reconstructed, not recovered.'}
    root=Path(args.cache)/args.device;root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'evaluation.lock'),timeout=0):
        path=root/'provenance.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=prov:raise ValueError('Provenance changed')
        atomic_json(prov,path);(root/'implementation.py').write_bytes(Path(__file__).read_bytes())
        qa_ids=[i for i,r in enumerate(items) if r['task']!='caption'];qa=[items[i] for i in qa_ids]
        objects={}
        def key(obj):return fingerprint({'object':obj,'provenance':prov})
        for r in items:
            for obj in (r['q'],r['d']):objects[key(obj)]=obj
            paths=r['q'].get('images',[])+r['d'].get('images',[])
            if len(paths)!=1 or hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()!=r['image_file_sha256']:
                raise ValueError('Image changed')
        for r in qa:
            obj={'text':r['q']['text']};objects[key(obj)]=obj
        model=None;start=time.perf_counter();done=0
        for k,obj in objects.items():
            p=root/'vectors'/f'{k}.pt'
            if p.exists():
                saved=torch.load(p,weights_only=True,map_location='cpu')
                if saved['fingerprint']!=k:raise ValueError('Cache mismatch')
                continue
            if model is None:
                model,bundle=load_bundle_encoder(args.checkpoint)
                if bundle['step']!=0:raise ValueError('Requires reconstructed initial bundle')
            with torch.no_grad():
                u,_=model.extract_features(obj)
                raw=F.normalize(F.normalize(u.float(),dim=-1).flatten(),dim=-1).cpu()
                initial=model.embedding_head(u).embedding.cpu()
            if not torch.isfinite(raw).all() or not torch.isfinite(initial).all():raise ValueError('Nonfinite vectors')
            atomic_save({'raw':raw,'step0_reconstructed':initial,'fingerprint':k},p)
            done+=1
            if done%25==0 or done==1:
                state={'state':'encoding','new_objects':done,'total_unique_objects':len(objects),'pid':os.getpid(),
                       'seconds':time.perf_counter()-start}
                atomic_json(state,root/'status.json');print(json.dumps(state),flush=True)
        def get(obj,kind):return torch.load(root/'vectors'/f'{key(obj)}.pt',weights_only=True,map_location='cpu')[kind]
        results={}
        for kind in ['raw','step0_reconstructed']:
            q=torch.stack([get(r['q'],kind) for r in items]);d=torch.stack([get(r['d'],kind) for r in items])
            qt=torch.stack([get({'text':r['q']['text']},kind) for r in qa])
            full=score_vectors(items,q,d);qa_full=score_vectors(qa,q[qa_ids],d[qa_ids]);text_only=score_vectors(qa,qt,d[qa_ids])
            results[kind]={'provenance':prov,'full':full,'qa_full':qa_full,'qa_text_only':text_only,
                           'qa_paired_delta':paired_delta(qa,qa_full,text_only)}
            atomic_json(results[kind],Path('artifacts/results/v5/validation_controls')/f'{kind}_{args.device}.json')
        atomic_json({'state':'completed','unique_objects':len(objects),
                     'scores':{k:v['full']['macro_r1'] for k,v in results.items()}},root/'status.json')
        print(json.dumps({k:v['full']['macro_r1'] for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
