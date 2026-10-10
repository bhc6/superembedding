"""Evaluate the validation-selected LoRA export once; cache test embeddings atomically."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0')
os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from superembedding import OptionEmbeddingHead
from superembedding.omnijev import load_omnijev,OmniJevEmbeddingEncoder
from superembedding.checkpoint import atomic_save,fingerprint
from scripts.experiment_suite import evaluate


def main():
    root=Path('data/fashion2400'); output=Path('runs/fashion2400/lora_raw')
    meta=json.loads((root/'metadata.json').read_text())
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    records=[r for r in records if r['split']=='test']
    head,basis,source=OptionEmbeddingHead.load(output/'embedding_head.pt')
    model=load_omnijev(source['decision_checkpoint'],source['base_model'],max_pixels=65536)
    encoder=OmniJevEmbeddingEncoder(model,basis)
    encoder.load_head(output/'embedding_head.pt'); encoder.eval()
    reference=torch.load(output/'reload_reference.pt',weights_only=True)
    reloaded=encoder.encode({'images':[reference['image']]})
    delta=(reloaded.embedding.cpu()-reference['embedding']).abs().max().item()
    if delta>1e-5: raise RuntimeError(f'Export reload mismatch: {delta}')
    identity=fingerprint({'config':head.config(),'source':source,
        'training':json.loads((output/'training.json').read_text()),'records':records})
    cache=output/'test_vectors';cache.mkdir(exist_ok=True)
    manifest=cache/'fingerprint.txt'
    if manifest.exists() and manifest.read_text()!=identity: raise ValueError('test cache identity mismatch')
    manifest.write_text(identity)
    slots=[];vectors=[]; start=time.perf_counter()
    for i,r in enumerate(records):
        target=cache/f"{r['id']}.pt"
        if target.exists(): saved=torch.load(target,weights_only=True)
        else:
            result=encoder.encode({'images':[str(root/r['image'])]})
            saved={'slots':result.slots.cpu(),'embedding':result.embedding.cpu()}
            atomic_save(saved,target)
        slots.append(saved['slots']);vectors.append(saved['embedding'])
        if i%50==0:print(json.dumps({'done':i+1,'total':len(records)}),flush=True)
    keys=meta['label_keys']
    maps={k:{v:i for i,v in enumerate(sorted({r['labels'][k] for r in records}))} for k in keys}
    labels=torch.tensor([[maps[k][r['labels'][k]] for k in keys] for r in records])
    metrics=evaluate(torch.stack(slots),torch.stack(vectors),labels,keys)
    report={'test':metrics,'reload_max_abs_diff':delta,'seconds':time.perf_counter()-start,
            'scope':'One seed, 100 LoRA updates; exploratory follow-up, not a matched multi-seed architecture comparison.'}
    (output/'test_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
