"""CPU optimizer/RNG continuation check through the real ablation trainer."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_general_ablation import run_one
from scripts.migrate_general_validation import tensor_digest
from superembedding.checkpoint import atomic_json
import torch


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    generator=torch.Generator().manual_seed(515)
    rows=[{'id':str(i),'source':'a' if i%2 else 'b','split':'train' if i<80 else 'val','key':str(i%7)} for i in range(96)]
    q=torch.randn(96,4,1024,generator=generator);d=torch.randn(96,4,1024,generator=generator)
    bank=({'test':'synthetic optimizer continuation only'},rows,q,d,'synthetic-fixed-seed-515')
    root=Path('C:/Users/123/.cache/superembedding/v5/ablation_resume_audit')
    full=root/'continuous';resumed=root/'interrupted'
    if (full/'last.pt').exists() or (resumed/'last.pt').exists():raise ValueError('Audit directories already exist; preserve them')
    run_one(bank,full,'output_small',17,1e-4,6)
    run_one(bank,resumed,'output_small',17,1e-4,6,stop_after=3)
    paused=torch.load(resumed/'last.pt',weights_only=True)
    if paused['step']!=3:raise ValueError('Expected checkpoint at step 3')
    run_one(bank,resumed,'output_small',17,1e-4,6)
    a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
    keys=['model','optimizer','rng_local','rng_python','rng_torch','rng_cuda','step','extra']
    ah=tensor_digest({k:a[k] for k in keys});bh=tensor_digest({k:b[k] for k in keys})
    if ah!=bh:raise ValueError('Continuation does not match uninterrupted optimizer/RNG/history')
    result={'passed':True,'continuous_digest':ah,'resumed_digest':bh,'stop_step':3,'final_step':6,
        'architecture':'output_small','scope':'Real trainer on synthetic 1024d features; not a retrieval quality experiment'}
    atomic_json(result,'artifacts/results/v5/general_ablation_resume_audit.json');print(json.dumps(result))


if __name__=='__main__':main()
