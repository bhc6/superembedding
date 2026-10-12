"""Complete missing precision-matched raw ScienceQA controls on CPU."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json


def main():
    root=Path(__file__).resolve().parents[1]
    out=Path('D:/DEMO/superembedding-storage/superembedding/v5/external_control_completion');out.mkdir(exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/'pipeline.lock'),timeout=0):
        script='scripts/evaluate_mmeb_scienceqa.py'
        seal={'protocol':'step250-raw-external-precision-control-v1','script_sha256':hashlib.sha256(Path(script).read_bytes()).hexdigest(),
              'backend':'omnijev_raw','device':'cpu','dtype':'float32','tasks':['full','text_only'],
              'checkpoint_selection':'none; frozen original OmniJev','test_based_tuning':False}
        path=out/'protocol.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=seal:raise ValueError('Control protocol changed')
        atomic_json(seal,path)
        commands=[[script,'--backend','omnijev_raw','--device','cpu','--query-mode',mode] for mode in seal['tasks']]
        commands += [['scripts/audit_experiment_completeness.py'],['scripts/report_validation_controls.py']]
        for i,command in enumerate(commands):
            if hashlib.sha256(Path(script).read_bytes()).hexdigest()!=seal['script_sha256']:raise ValueError('Evaluator changed')
            atomic_json({'state':'running','stage':i,'command':command,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat()},out/'status.json')
            with (out/f'stage_{i}.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u',*command],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json({'state':'failed','stage':i,'exit_code':result.returncode},out/'status.json')
                raise RuntimeError('Control completion failed; preserve cache and rerun after inspection')
        atomic_json({'state':'completed','stages':len(commands),'updated_utc':datetime.now(timezone.utc).isoformat()},out/'status.json')


if __name__=='__main__':main()
