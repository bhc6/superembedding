"""Serialize memory-heavy official baselines while GPU training continues."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superembedding.checkpoint import atomic_json
from superembedding.leakage_guard import digest
OUT = Path('D:/DEMO/superembedding-storage/superembedding/v5/paper_cpu_queue_v1')
EVALUATOR = ROOT/'scripts/evaluate_paper_baseline_cpu.py'


def main():
    from filelock import FileLock
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    plan = {'models': ['wemm2b', 'gme2b'], 'evaluator_sha256': digest(EVALUATOR),
            'queue_sha256': digest(__file__), 'resource': 'CPU FP32, two threads, one model at a time',
            'scope': 'Seen diagnostics only; fixed official weights, restricted pixels, not headline reproduction'}
    with FileLock(str(OUT/'queue.lock'), timeout=0):
        path = OUT/'protocol.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Queue protocol changed')
        atomic_json(plan, path)
        for model in plan['models']:
            if (OUT/'STOP').exists():
                atomic_json({'state': 'paused_before_model', 'model': model}, OUT/'status.json')
                return
            if digest(EVALUATOR) != plan['evaluator_sha256']:
                raise ValueError('Evaluator changed')
            status = Path('D:/DEMO/superembedding-storage/superembedding/v5/paper_baselines_cpu_v3')/model/'status.json'
            if status.exists() and json.loads(status.read_text()).get('state') == 'completed':
                continue
            with (OUT/(model+'.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(EVALUATOR), '--model', model],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                atomic_json({'state': 'running', 'model': model, 'pid': os.getpid(), 'child_pid': child.pid,
                             'utc': datetime.now(timezone.utc).isoformat(), 'log': str(OUT/(model+'.log'))}, OUT/'status.json')
                while child.poll() is None:
                    time.sleep(15)
                code = child.returncode
            detail = json.loads(status.read_text()) if status.exists() else {}
            if code or detail.get('state') != 'completed':
                atomic_json({'state': 'failed_or_paused', 'model': model, 'exit_code': code,
                             'worker': detail, 'resume': 'Inspect error and rerun; cached vectors retained'}, OUT/'status.json')
                return
        atomic_json({'state': 'completed', 'models': plan['models']}, OUT/'status.json')


if __name__ == '__main__':
    main()
