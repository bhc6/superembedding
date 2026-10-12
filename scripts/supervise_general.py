"""Finite detached training with bounded retries and checkpoint validation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=2000);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((Path(args.data)/'audit.json').read_text())
    import hashlib
    if hashlib.sha256((Path(args.data)/'records.jsonl').read_bytes()).hexdigest()!=audit['manifest_sha256']:
        raise RuntimeError('Data audit stale')
    command=[sys.executable,'-u','scripts/train_general.py','--data',args.data,'--output',args.output,
             '--steps',str(args.steps),'--batch-size','16','--checkpoint-every','5',
             '--validate-every','250','--milestone-every','500']
    info={'supervisor_pid':os.getpid(),'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),
          'max_restarts':2,'data_manifest_sha256':audit['manifest_sha256']}
    for attempt in range(3):
        with (out/f'worker_attempt_{attempt}.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            atomic_json({**info,'attempt':attempt,'worker_pid':process.pid,'state':'running'},out/'supervisor.json')
            code=process.wait()
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if code==0:
            if status.get('state')=='completed':
                atomic_json({**info,'attempt':attempt,'state':'final_evaluation','exit_code':code},out/'supervisor.json')
                with (out/'final_evaluation_driver.log').open('a',encoding='utf-8') as log:
                    evaluated=subprocess.run([sys.executable,'-u','scripts/finish_general_stage.py','--run',args.output],
                                             cwd=root,stdout=log,stderr=subprocess.STDOUT)
                atomic_json({**info,'attempt':attempt,'state':'stage_complete' if evaluated.returncode==0 else 'evaluation_failed',
                             'exit_code':evaluated.returncode},out/'supervisor.json')
            else:atomic_json({**info,'attempt':attempt,'state':status.get('state','exited'),'exit_code':code},out/'supervisor.json')
            return
        # Deterministic/data/disk failures need a fix, not automatic repeated runs.
        reason=str(status.get('message',''))
        if any(s in reason for s in ['MiB free','fingerprint','Nonfinite','mismatch','Insufficient','No negative']):break
        if not (out/'last.pt').exists():break
        time.sleep(15)
    atomic_json({**info,'state':'failed','last_status':status,'exit_code':code},out/'supervisor.json')
    raise RuntimeError('Training stopped; see preserved worker logs and checkpoint')

if __name__=='__main__':main()
