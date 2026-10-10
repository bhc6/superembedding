"""Fresh image-to-vector checks for all backends/domains and compressed heads."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import OptionEmbeddingHead

ROOT=Path(__file__).resolve().parents[1]

def main():
    seal=json.loads((ROOT/'artifacts/results/v3/selection_sealed.json').read_text())
    selected=[r for r in seal['selected'] if r['scale']==100 and r['seed']==42
              and (r['arch']=='raw' or r['backend']=='omnijev' and r['arch']=='joint_out_small')]
    assert len(selected)==12
    out=ROOT/'runs/v3/inference';out.mkdir(exist_ok=True);results=[]
    for item in selected:
        dataset,backend=item['dataset'],item['backend']; data=ROOT/'data'/dataset
        records=[json.loads(s) for s in (data/'records.jsonl').read_text().splitlines()]
        record=next(r for r in records if r['split']=='train')
        headpath=ROOT/item['path']/'best_head.pt'; name=f'{dataset}_{backend}_{item["arch"]}'
        target=out/f'{name}.pt'
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'scripts/encode_research_image.py','--head',str(headpath),
                '--image',str((data/record['image']).resolve()),'--output',str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=torch.load(target,map_location='cpu',weights_only=True)
        cached=torch.load(ROOT/'runs/v3'/dataset/backend/f"{record['id']}.pt",map_location='cpu',weights_only=True)
        head,basis,_=OptionEmbeddingHead.load(headpath);head.eval()
        with torch.no_grad(): expected=head(cached['u'],cached['zq'])
        assert actual['directions']==list(basis.directions) and actual['backend']==backend
        row={'dataset':dataset,'backend':backend,'arch':item['arch'],'image_id':record['id']}
        for key in ('slots','embedding'):
            reference=getattr(expected,key);vector=actual[key]
            assert vector.shape==reference.shape and torch.isfinite(vector).all()
            assert torch.allclose(vector.norm(dim=-1),torch.ones_like(vector[...,0]),atol=1e-5)
            cosine=F.cosine_similarity(vector,reference,dim=-1).min().item()
            if cosine<.99: raise AssertionError(f'Fresh inference discrepancy: {name} {key} {cosine}')
            row[key+'_minimum_cosine_to_cached']=cosine;row[key+'_max_abs_difference']=(vector-reference).abs().max().item()
        results.append(row);print(json.dumps(row),flush=True)
    (ROOT/'artifacts/results/v3/fresh_inference_verification.json').write_text(json.dumps({'checks':results,
        'minimum_accepted_cosine':.99,'note':'BF16 batch-size and CPU/GPU projection differences are expected; this is not a bitwise equivalence assertion.'},indent=2))

if __name__=='__main__': main()
