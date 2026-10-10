"""Start an independent domain's heads when frozen feature extraction finishes."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['cub_attributes','dtd']);args=ap.parse_args()
    for backend in ('omnijev','base','flair'):
        while not (ROOT/'runs/v3'/args.dataset/backend/'extraction.json').exists(): time.sleep(10)
        log=ROOT/'runs'/f'v3_parallel_train_{args.dataset}_{backend}.log'
        with log.open('a',encoding='utf-8') as output:
            subprocess.run([sys.executable,'scripts/research_suite.py','--dataset',args.dataset,'--backend',backend],
                           cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
