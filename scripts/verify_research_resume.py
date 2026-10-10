"""Run actual v3 training uninterrupted vs interrupted; compare model/optimizer/RNG."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import torch

def main():
    root=Path(__file__).resolve().parents[1]
    data=root/'data/v3_resume_audit'; data.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'data/fashion_expanded/metadata.json').read_text())
    rows=[json.loads(s) for s in (root/'data/fashion_expanded/records.jsonl').read_text().splitlines()]
    cache=root/'runs/v3/fashion_expanded/omnijev'
    subset=[]
    for split,count in [('train',160),('val',40),('test',40)]:
        subset.extend([r for r in rows if r['split']==split and (cache/f"{r['id']}.pt").exists()][:count])
    counts={s:sum(r['split']==s for r in subset) for s in ('train','val','test')}
    if min(counts.values())<20: raise RuntimeError(f'Need more features for audit {counts}')
    meta['scales']={'100':counts['train']}
    (data/'metadata.json').write_text(json.dumps(meta)); (data/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset))
    audit_id=hashlib.sha256(json.dumps(subset,sort_keys=True).encode()+(cache/'provenance.json').read_bytes()
        +(root/'scripts/research_suite.py').read_bytes()).hexdigest()[:16]
    audit_root=root/'runs/v3_resume'/audit_id
    common=[sys.executable,'scripts/research_suite.py','--dataset','v3_resume_audit','--backend','omnijev',
        '--data',str(data),'--features',str(cache),'--scales','100','--seeds','42','--lrs','.001',
        '--steps','4','--architectures','joint_out_small']
    for output,extra in [('continuous',[]),('resumed',['--stop-after','2']),('resumed',[])]:
        subprocess.run(common+['--output-root',str(audit_root/output)]+extra,cwd=root,check=True)
    path='v3_resume_audit/suite/omnijev/n100/joint_out_small_s42_lr0.001/last.pt'
    a=torch.load(audit_root/'continuous'/path,map_location='cpu',weights_only=True)
    b=torch.load(audit_root/'resumed'/path,map_location='cpu',weights_only=True)
    def compare(x,y):
        if isinstance(x,torch.Tensor): return torch.equal(x,y)
        if isinstance(x,dict): return x.keys()==y.keys() and all(compare(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)): return len(x)==len(y) and all(compare(i,j) for i,j in zip(x,y))
        return x==y
    result={k:compare(a[k],b[k]) for k in ('model','optimizer','rng_local','rng_python','rng_torch','rng_cuda','config','step','extra')}
    result['model_max_abs_difference']=max((a['model'][k]-b['model'][k]).abs().max().item() for k in a['model'])
    if not all(v for k,v in result.items() if k!='model_max_abs_difference'): raise AssertionError(result)
    result['audit_id']=audit_id
    out=root/'artifacts/results/v3'; out.mkdir(parents=True,exist_ok=True)
    (out/'resume_verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
