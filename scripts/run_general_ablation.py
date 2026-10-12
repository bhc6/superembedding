"""Finite CPU pipeline alongside GPU training; no publication actions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--root',required=True)
    a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    features=root.parent/'general_ablation_features';experiments=root/'experiments'
    commands=[['scripts/extract_general_ablation.py','--data',a.data,'--output',str(features)],
        ['scripts/train_general_ablation.py','--features',str(features),'--output',str(experiments)],
        ['scripts/evaluate_general_ablation.py','--features',str(features),'--output',str(experiments)]]
    for i,command in enumerate(commands):
        atomic_json({'state':'running','stage':i,'command':command},root/'pipeline_status.json')
        with (root/f'stage_{i}.log').open('a',encoding='utf-8') as log:
            process=subprocess.run([sys.executable,'-u',*command],cwd=Path(__file__).resolve().parents[1],
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        if process.returncode:
            atomic_json({'state':'failed','stage':i,'exit_code':process.returncode},root/'pipeline_status.json')
            raise RuntimeError('CPU ablation stage failed; caches and checkpoints preserved')
        if i==0 and json.loads((features/'status.json').read_text()).get('state')!='completed':
            atomic_json({'state':'paused','stage':i},root/'pipeline_status.json');return
    atomic_json({'state':'completed','stages':len(commands),'publication':'not submitted'},root/'pipeline_status.json')


if __name__=='__main__':
    from filelock import FileLock
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--root',required=True)
    a,_=parser.parse_known_args();Path(a.root).mkdir(parents=True,exist_ok=True)
    with FileLock(str(Path(a.root)/'pipeline.lock'),timeout=0):main()
