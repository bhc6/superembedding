"""Finish export/audit/artifacts after the running experiment matrix completes."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    while not (ROOT/'runs/fashion2400/lora_raw/test_results.json').exists():
        time.sleep(10)
    for script in ['export_lora_last','verify_exports','summarize_results','audit_release','plot_results','package_release']:
        print('START',script,flush=True)
        subprocess.run([sys.executable,f'scripts/{script}.py'],cwd=ROOT,check=True)
    print('Local release artifacts complete; GitHub publication is a separate explicit invocation.',flush=True)
