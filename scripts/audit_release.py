"""Evidence checks before publication. Does not modify experimental outcomes."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding import OptionEmbeddingHead

ROOT=Path(__file__).resolve().parents[1]


def main():
    from superembedding.leakage_guard import require_promotion_clearance
    require_promotion_clearance()
    root=ROOT/'data/fashion2400'
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in (ROOT/'data/fashion600/records.jsonl').read_text().splitlines()]
    assert len(records)==2400
    assert len({r['id'] for r in records})==len(records)
    assert len({r['image_sha256'] for r in records})==len(records)
    assert not ({r['id'] for r in records}&{r['id'] for r in old})
    assert not ({r['image_sha256'] for r in records}&{r['image_sha256'] for r in old})
    ph=[int(r['dhash'],16) for r in records]
    minimum=min((a^b).bit_count() for i,a in enumerate(ph) for b in ph[:i])
    assert minimum>4
    runs=[]
    for suite in ['suite','suite_base','suite_flair']:
        for p in (ROOT/'runs/fashion2400'/suite).glob('*/validation.json'):
            report=json.loads(p.read_text());folder=p.parent
            last=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
            assert last['step']==500 and report['test_evaluated'] is False
            head,basis,_=OptionEmbeddingHead.load(folder/'best_head.pt')
            assert all(torch.isfinite(t).all() for t in head.state_dict().values())
            assert list(basis.directions)==json.loads((root/'metadata.json').read_text())['directions']
            runs.append({'suite':suite,'run':folder.name,'step':last['step'],'best_step':report['best_step']})
    assert len(runs)==54
    lora=ROOT/'runs/fashion2400/lora_raw'
    last=torch.load(lora/'last.pt',weights_only=True,map_location='cpu')
    assert last['step']==100 and last['partial']
    train=json.loads((lora/'training.json').read_text())
    assert len([r for r in train['history'] if 'loss' in r])==100
    assert '"resume_step": 10' in (ROOT/'runs/lora_formal.log').read_text()
    test=json.loads((lora/'test_results.json').read_text())
    assert test['reload_max_abs_diff']<=1e-5
    audit={'dataset_count':len(records),'min_pairwise_dhash_distance':minimum,
           'development_overlap':0,'completed_cached_runs':len(runs),'completed_lora_steps':100,
           'lora_resumed_at_step':10,'reload_max_abs_diff':test['reload_max_abs_diff'],'runs':runs}
    (ROOT/'artifacts/results/release_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
