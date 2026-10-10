"""Compact local status; reads no test metrics and does not mutate the queue."""
import json
from pathlib import Path
import time

def main():
    report={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'datasets':{}}
    for dataset in ('fashion_expanded','cub_attributes','dtd'):
        path=Path('data')/dataset/'metadata.json'
        if not path.exists(): continue
        meta=json.loads(path.read_text()); status={'images':meta['count'],'features':{},'completed_training':{}}
        for backend in ('omnijev','base','flair'):
            root=Path('runs/v3')/dataset
            status['features'][backend]=sum(1 for p in (root/backend).glob('*.pt'))
            status['completed_training'][backend]=sum(1 for p in (root/'suite'/backend).glob('*/*/validation.json'))
        report['datasets'][dataset]=status
    report['complete_training_runs']=sum(sum(v['completed_training'].values()) for v in report['datasets'].values())
    report['expected_training_runs']=432
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
