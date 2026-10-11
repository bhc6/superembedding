"""Wait for the complete grid, then run audits and reports; never publish automatically."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--model-cache',default='models/v4');a=ap.parse_args()
    while len(list(Path('runs/v4').glob('*/suite/*/*/*/validation.json')))<324:time.sleep(10)
    commands=[
        ['scripts/verify_baseline_resume.py','--features',a.features,'--backend','qwen3_vl_instruct'],
        ['scripts/evaluate_baselines.py','--features',a.features],
        ['scripts/verify_baseline_inference.py','--features',a.features,'--model-cache',a.model_cache],
        ['scripts/account_baseline_api.py','--features',a.features],
        ['scripts/report_baselines.py'],
    ]
    for cmd in commands:
        print('RUN '+' '.join(cmd),flush=True)
        subprocess.run([sys.executable,*cmd],check=True)
    print('V4 evaluation and reporting complete',flush=True)

if __name__=='__main__':main()
