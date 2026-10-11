"""Exercise the actual packed-feature trainer: uninterrupted 4 vs resumed 2+2."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.baselines import read_data
from superembedding.checkpoint import fingerprint,atomic_save,atomic_json

def equal(x,y):
    if isinstance(x,torch.Tensor):return torch.equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
    return x==y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--backend',default='clip_b32');a=ap.parse_args()
    _,meta,rows=read_data('fashion_expanded');cache=Path(a.features)/'fashion_expanded'/a.backend
    selected=[i for split,n in [('train',160),('val',40),('test',40)] for i in [i for i,r in enumerate(rows) if r['split']==split][:n]]
    records=[rows[i] for i in selected];meta['scales']={'100':160}
    prov=json.loads((cache/'provenance.json').read_text());prov['record_fingerprint']=fingerprint(records)
    root=Path('runs/v4/resume_audit')/a.backend;data=root/'data';data.mkdir(parents=True,exist_ok=True)
    atomic_json(meta,data/'metadata.json');(data/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    small=root/'features';atomic_json(prov,small/'provenance.json')
    p=torch.load(cache/'features.pt',weights_only=True)
    p.update(ids=[r['id'] for r in records],hashes=[r['image_sha256'] for r in records],vectors=p['vectors'][selected],provenance_fingerprint=fingerprint(prov))
    atomic_save(p,small/'features.pt')
    common=[sys.executable,'scripts/research_suite.py','--dataset','audit','--backend',a.backend,'--protocol','v4',
        '--data',str(data),'--features',str(small),'--scales','100','--seeds','42','--lrs','.001','--steps','4','--architectures','raw']
    for name,extra in [('continuous',[]),('resumed',['--stop-after','2']),('resumed',[])]:
        subprocess.run(common+['--output-root',str(root/name)]+extra,check=True)
    relative=Path('audit/suite')/a.backend/'n100/raw_s42_lr0.001/last.pt'
    x=torch.load(root/'continuous'/relative,weights_only=True,map_location='cpu')
    y=torch.load(root/'resumed'/relative,weights_only=True,map_location='cpu')
    result={k:equal(x[k],y[k]) for k in ('model','optimizer','rng_local','rng_python','rng_torch','rng_cuda','config','step','extra')}
    assert all(result.values()),result
    result['max_model_abs_difference']=max((x['model'][k]-y['model'][k]).abs().max().item() for k in x['model'])
    atomic_json(result,Path('artifacts/results/v4')/f'resume_{a.backend}.json');print(json.dumps(result))

if __name__=='__main__':main()
