"""After all training finishes, audit, seal selection, evaluate, and report locally.

No Git commit, push, or release publication occurs here.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
from filelock import FileLock

ROOT=Path(__file__).resolve().parents[1]

def run(script):
    log=ROOT/'runs/v3/logs'/f'finalize_{Path(script).stem}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    print(f'START {script}',flush=True)
    with log.open('a',encoding='utf-8') as out:
        subprocess.run([sys.executable,script],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    print(f'DONE {script}',flush=True)

if __name__=='__main__':
    with FileLock(str(ROOT/'runs/v3/finalize.lock'),timeout=0):
        while len(list((ROOT/'runs/v3').glob('*/suite/*/*/*/validation.json')))<432: time.sleep(15)
        for script in ('verify_research_resume','audit_research','verify_research_exports','evaluate_research','report_research','verify_research_inference'):
            run('scripts/'+script+'.py')
        resources=ROOT/'runs/v3/resources.jsonl'
        if resources.exists():
            samples=[json.loads(s) for s in resources.read_text().splitlines()]
            report={'sample_count':len(samples),'sampling_seconds':15,'first':samples[0]['timestamp'],'last':samples[-1]['timestamp'],
                    'peak_total_GPU_memory_MiB':max(r['memory_MiB'] for r in samples),
                    'peak_temperature_C':max(r['temperature_C'] for r in samples),
                    'peak_power_W':max(r['power_W'] for r in samples),
                    'caveat':'Periodic device-level observations can miss shorter peaks; include concurrent processes, not only this project.'}
            (ROOT/'artifacts/results/v3/resource_observations.json').write_text(json.dumps(report,indent=2))
        (ROOT/'runs/v3/local_evaluation_complete.json').write_text(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'test_complete':True,'published':False},indent=2))
        print('Local evaluation complete; review reports and inference exports before packaging/publication.',flush=True)
