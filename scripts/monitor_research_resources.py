"""Local evidence for 8GB resource claims; no network traffic or notifications."""
import json
from pathlib import Path
import subprocess
import time

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]; log=root/'runs/v3/resources.jsonl'
    while not (root/'runs/v3/stop_resource_monitor').exists():
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True)
        if result.returncode==0:
            parts=result.stdout.strip().split(','); values=list(map(float,parts))
            item=dict(zip(('memory_MiB','utilization_percent','power_W','temperature_C'),values)); item['timestamp']=time.strftime('%Y-%m-%d %H:%M:%S')
            with log.open('a') as out: out.write(json.dumps(item)+'\n')
        time.sleep(15)
