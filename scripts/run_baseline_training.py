"""Resume the predeclared v4 training grid as feature banks become available."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.baselines import DATASETS,MODELS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True)
    ap.add_argument('--root',default='runs/v4');a=ap.parse_args()
    root=Path(a.root);pending=[(d,b) for b in MODELS for d in DATASETS]
    while pending:
        available=[(d,b) for d,b in pending if (Path(a.features)/d/b/'features.pt').exists()]
        if not available:time.sleep(10);continue
        for d,b in available:
            log=root/'logs'/f'train_{d}_{b}.log';log.parent.mkdir(parents=True,exist_ok=True)
            print(f'TRAIN {d} {b}',flush=True)
            with log.open('a',encoding='utf-8') as stream:
                subprocess.run([sys.executable,'scripts/research_suite.py','--dataset',d,'--backend',b,
                    '--protocol','v4','--features',str(Path(a.features)/d/b),'--output-root',str(root),
                    '--architectures','raw'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            pending.remove((d,b));print(f'COMPLETE {d} {b}; remaining={len(pending)}',flush=True)
    print('All 324 training runs complete',flush=True)

if __name__=='__main__':main()
