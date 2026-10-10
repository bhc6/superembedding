"""Benchmark only training images, compare the official single-image branch path."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0'); os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from torch.nn import functional as F
from superembedding import AttributeBasis
from superembedding.omnijev import load_omnijev,OmniJevEmbeddingEncoder
from superembedding.batched import prepare_image,encode_prepared

def main():
    torch.set_num_threads(4)
    root=Path('data/fashion_expanded'); meta=json.loads((root/'metadata.json').read_text())
    rows=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
    encoder=OmniJevEmbeddingEncoder(mso,AttributeBasis(tuple(meta['directions']),meta['instruction'])).eval()
    groups={}; records={}
    for r in rows:
        if r['split']!='train': continue
        prepared=prepare_image(encoder,root/r['image']); key=prepared['key']
        groups.setdefault(key,[]).append(prepared); records.setdefault(key,[]).append(r)
        if len(groups[key])==4: break
    items=groups[key]; selected=records[key]
    reference=[]; start=time.perf_counter()
    for r in selected: reference.append(encoder.extract_features({'images':[str(root/r['image'])]}))
    torch.cuda.synchronize(); single=(time.perf_counter()-start)/len(items)
    refu=torch.stack([x[0] for x in reference]); refq=torch.stack([x[1] for x in reference]); report=[]
    for batch in (1,2,4):
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        for _ in range(3): u,zq=encode_prepared(encoder,items[:batch])
        torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)/(3*batch)
        report.append({'batch':batch,'forward_seconds_per_image':elapsed,'single_seconds_per_image':single,
            'u_max_abs_diff':(u-refu[:batch]).abs().max().item(),'zq_max_abs_diff':(zq-refq[:batch]).abs().max().item(),
            'u_min_cosine':F.cosine_similarity(u,refu[:batch],dim=-1).min().item(),
            'zq_min_cosine':F.cosine_similarity(zq,refq[:batch],dim=-1).min().item(),
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        print(json.dumps(report[-1]),flush=True)
    out=Path('artifacts/results/v3/batch_benchmark.json'); out.write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
