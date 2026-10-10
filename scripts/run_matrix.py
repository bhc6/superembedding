"""Sequential GPU experiment ledger; safe to restart after any completed stage."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def run(name,arguments):
    ledger=ROOT/'runs/fashion2400/ledger.jsonl'
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('a',encoding='utf-8') as f: f.write(json.dumps({'stage':name,'event':'start','utc':time.time(),'arguments':arguments})+'\n')
    with (ROOT/'runs'/f'{name}.log').open('a',encoding='utf-8') as log:
        proc=subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    with ledger.open('a',encoding='utf-8') as f: f.write(json.dumps({'stage':name,'event':'finish','utc':time.time(),'returncode':proc.returncode})+'\n')
    if proc.returncode: raise RuntimeError(f'{name} failed: runs/{name}.log')
    print(name,'complete',flush=True)


def main():
    # The first extraction may already be running when this coordinator starts.
    while not (ROOT/'runs/fashion2400/omnijev/extraction.json').exists(): time.sleep(10)
    run('matrix_omnijev',['scripts/experiment_suite.py'])
    run('extract_base',['scripts/extract_features.py','--data','data/fashion2400','--output','runs/fashion2400/base','--disable-lora'])
    run('matrix_base',['scripts/experiment_suite.py','--features','runs/fashion2400/base','--output','runs/fashion2400/suite_base','--architectures','raw'])
    run('extract_flair_matrix',['scripts/extract_flair.py'])
    run('matrix_flair',['scripts/experiment_suite.py','--features','runs/fashion2400/flair','--output','runs/fashion2400/suite_flair','--architectures','raw'])
    # LoRA initialization is chosen solely from frozen-head validation.
    candidates=[]
    for p in (ROOT/'runs/fashion2400/suite').glob('raw_s42_*/validation.json'):
        r=json.loads(p.read_text()); candidates.append((r['best'],p.parent/'best_head.pt'))
    _,head=max(candidates)
    lora=ROOT/'runs/fashion2400/lora_raw'
    if not (lora/'training.json').exists():
        arguments=['scripts/train_lora.py','--data','data/fashion2400','--head',str(head),'--output',str(lora),
                          '--steps','100','--accumulation','1','--lr','1e-5','--margin','0.2','--seed','42',
                          '--checkpoint-every','10','--validate-every','25']
        if not (lora/'last.pt').exists():
            run('lora_pause_test',arguments+['--stop-after','10'])
        run('lora_formal',arguments)
    for suffix,features,suite in [('omnijev','omnijev','suite'),('base','base','suite_base'),('flair','flair','suite_flair')]:
        if (ROOT/f'runs/fashion2400/results_{suffix}.json').exists():
            continue
        run('evaluate_'+suffix,['scripts/evaluate_suite.py','--features',f'runs/fashion2400/{features}',
            '--suite',f'runs/fashion2400/{suite}','--output',f'runs/fashion2400/results_{suffix}.json'])
    if not (lora/'test_results.json').exists():
        run('evaluate_lora',['scripts/evaluate_lora.py'])


if __name__=='__main__': main()
