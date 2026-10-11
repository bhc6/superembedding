"""Aggregate sanitized provider usage from successful resumable API shards."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.baselines import DATASETS,MODELS
from superembedding.checkpoint import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);a=ap.parse_args()
    results={}
    for b,spec in MODELS.items():
        if 'model' not in spec:continue
        counts=Counter();dates=[];request_ids=set()
        for d in DATASETS:
            for file in (Path(a.features)/d/b/'shards').glob('*.pt'):
                s=torch.load(file,weights_only=True);info=s['info'];usage=info['usage']
                if info.get('request_id'):
                    assert info['request_id'] not in request_ids
                    request_ids.add(info['request_id'])
                counts.update(requests=1,images=len(s['ids']),extra_attempts=info['attempts']-1)
                for key in ('total_tokens','input_tokens','image_tokens','output_tokens'):
                    counts[key]+=usage.get(key,0)
                for key,value in usage.get('input_tokens_details',{}).items():counts['detail_'+key]+=value
                dates.append(info['utc'])
        results[b]={**dict(counts),'first_utc':min(dates),'last_utc':max(dates)}
    atomic_json(results,'artifacts/results/v4/api_accounting.json');print(results)

if __name__=='__main__':main()
