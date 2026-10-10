"""Sequential GPU queue, safe to rerun; publication is intentionally separate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from filelock import FileLock

ROOT=Path(__file__).resolve().parents[1]

def command(args,name):
    log=ROOT/'runs/v3/logs'/f'{name}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'stage':name,'argv':args,'time':time.strftime('%Y-%m-%d %H:%M:%S')}),flush=True)
    with log.open('a',encoding='utf-8') as out:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--datasets',nargs='+',default=['fashion_expanded','cub_attributes','dtd'])
    ap.add_argument('--parallel',action='store_true',help='Up to three encoders and one cached-head trainer; tested on an 8GB GPU')
    args=ap.parse_args()
    (ROOT/'runs/v3').mkdir(parents=True,exist_ok=True)
    with FileLock(str(ROOT/'runs/v3/queue.lock'),timeout=0):
        def extract(dataset,backend):
            if dataset=='fashion_expanded' and backend=='flair': command(['scripts/reuse_pilot_features.py','--backend',backend],f'reuse_{backend}')
            script='scripts/extract_flair.py' if backend=='flair' else 'scripts/extract_batched.py'
            cli=[script,'--data',f'data/{dataset}','--output',f'runs/v3/{dataset}/{backend}']
            if backend=='base': cli+=['--disable-lora']
            command(cli,f'extract_{dataset}_{backend}')
            return dataset,backend
        def train(dataset,backend):
            command(['scripts/research_suite.py','--dataset',dataset,'--backend',backend],f'train_{dataset}_{backend}')
        if args.parallel:
            with ThreadPoolExecutor(max_workers=3) as encoders,ThreadPoolExecutor(max_workers=1) as trainers:
                futures=[encoders.submit(extract,d,b) for d in args.datasets for b in ('omnijev','base','flair')]
                training=[]
                for future in as_completed(futures): training.append(trainers.submit(train,*future.result()))
                for future in training: future.result()
            (ROOT/'runs/v3/training_complete.json').write_text(json.dumps({'datasets':args.datasets,'test_evaluated':False},indent=2))
            return
        for dataset in args.datasets:
            if not (ROOT/'data'/dataset/'metadata.json').exists():
                raise RuntimeError(f'Dataset not prepared: {dataset}')
            for backend in ('omnijev','base','flair'):
                extract(dataset,backend);train(dataset,backend)
        (ROOT/'runs/v3/training_complete.json').write_text(json.dumps({'datasets':args.datasets,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'test_evaluated':False},indent=2))
        print('All requested training completed. Run independent audit before unsealing test.',flush=True)

if __name__=='__main__': main()
