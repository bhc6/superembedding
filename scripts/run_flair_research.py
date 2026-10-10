"""Small-memory official FLAIR baseline queue, concurrent with frozen extraction."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def run(args,log):
    with (ROOT/'runs'/log).open('a',encoding='utf-8') as out:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    for dataset in ('fashion_expanded','cub_attributes','dtd'):
        while not (ROOT/'data'/dataset/'metadata.json').exists(): time.sleep(10)
        if dataset=='fashion_expanded': run(['scripts/reuse_pilot_features.py','--backend','flair'],'v3_parallel_flair.log')
        run(['scripts/extract_flair.py','--data',f'data/{dataset}','--output',f'runs/v3/{dataset}/flair'],f'v3_flair_extract_{dataset}.log')
        run(['scripts/research_suite.py','--dataset',dataset,'--backend','flair'],f'v3_flair_train_{dataset}.log')
