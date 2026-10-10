"""Bounded prefetch alongside the main queue; no more than three large encoders.

Invoke only with the documented initial workers (Fashion Omni, CUB Omni, CUB base).
File locks in extract_features make queue retries and overlapping requests safe.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def wait_file(path):
    while not (ROOT/path).exists(): time.sleep(10)

def run(args,log):
    with (ROOT/'runs'/log).open('a',encoding='utf-8') as out:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)

def fashion_base():
    wait_file('runs/v3/fashion_expanded/omnijev/extraction.json')
    run(['scripts/extract_batched.py','--data','data/fashion_expanded','--output','runs/v3/fashion_expanded/base','--disable-lora'],'v3_prefetch_fashion_base.log')

def dtd_features():
    wait_file('data/official/dtd_verified.json')
    if not (ROOT/'data/dtd/metadata.json').exists(): run(['scripts/prepare_multidomain.py','dtd'],'prepare_dtd.log')
    # The CUB Omni worker has released its GPU before this encoder starts.
    wait_file('runs/v3/cub_attributes/omnijev/extraction.json')
    for backend in ('omnijev','base'):
        args=['scripts/extract_batched.py','--data','data/dtd','--output',f'runs/v3/dtd/{backend}']
        if backend=='base': args+=['--disable-lora']
        run(args,f'v3_prefetch_dtd_{backend}.log')

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fashion_base),pool.submit(dtd_features)]
        for future in futures: future.result()
