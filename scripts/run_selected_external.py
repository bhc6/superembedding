"""Wait for priority controls, then frozen interim external evaluation on CPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json


def main():
    root=Path(__file__).resolve().parents[1]
    store=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    out=store/'selected_external_step250';out.mkdir(exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/'pipeline.lock'),timeout=0):
        bundle=store/'validation_selected_step250/general_model.pt'
        seal=json.loads((store/'priority_validation/selection_sealed.json').read_text(encoding='utf-8'))
        if hashlib.sha256(bundle.read_bytes()).hexdigest()!=seal['export_sha256']:raise ValueError('Selected bundle changed')
        atomic_json({**seal,'scope':'Interim Step250 diagnostic, no test-based checkpoint or architecture selection',
                     'tests':['Official MMEB ScienceQA 1000','Urban1k common999','Synthetic paired evidence v1']},out/'selection_sealed.json')
        deadline=time.monotonic()+8*3600
        waits=[store/'priority_validation/status.json',store/'validation_controls/qwen2b_cpu/status.json']
        while True:
            states=[json.loads(p.read_text(encoding='utf-8')) if p.exists() else {} for p in waits]
            if all(s.get('state')=='completed' for s in states):break
            if any(s.get('state','').startswith('failed') for s in states):raise RuntimeError('Priority prerequisite failed; external tests not started')
            if time.monotonic()>deadline:raise RuntimeError('Bounded wait expired; rerun after prerequisites complete')
            atomic_json({'state':'waiting_for_priority_controls','pid':os.getpid(),'prerequisite_states':[s.get('state') for s in states]},out/'status.json')
            time.sleep(15)
        commands=[
            ['scripts/report_validation_controls.py'],
            ['scripts/evaluate_mmeb_scienceqa.py','--backend','general','--checkpoint',str(bundle),'--device','cpu'],
            ['scripts/evaluate_mmeb_scienceqa.py','--backend','general','--checkpoint',str(bundle),'--device','cpu','--query-mode','text_only'],
            ['scripts/evaluate_general_transfer.py','--checkpoint',str(bundle),'--device','cpu'],
            ['scripts/evaluate_general_transfer.py','--device','cpu'],
            ['scripts/report_selected_external.py'],
        ]
        for i,command in enumerate(commands):
            atomic_json({'state':'evaluating','stage':i,'command':command,**seal},out/'status.json')
            with (out/f'stage_{i}.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u',*command],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json({'state':'failed','stage':i,'exit_code':result.returncode,**seal},out/'status.json')
                raise RuntimeError('External evaluation failed; preserve vectors and inspect stage log')
        atomic_json({'state':'completed','stages':len(commands),**seal},out/'status.json')


if __name__=='__main__':main()
