"""Fetch pinned upstream code without vendoring it into this project."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPOS={
    'OmniJev':('https://github.com/tinnel123666888/OmniJev.git','14dbec4f71e194852c8d7b88ab36ef639493f400'),
    'FLAIR':('https://github.com/ExplainableML/flair.git','0f9ce05886d1ee0aa2c628653104cce77a1fff14'),
}

if __name__=='__main__':
    for name,(url,commit) in REPOS.items():
        folder=ROOT/'repos'/name
        if not (folder/'.git').exists():
            subprocess.run(['git','clone','--filter=blob:none',url,str(folder)],check=True)
        current=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        if current!=commit:
            dirty=subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True)
            if dirty: raise RuntimeError(f'Refusing to change modified upstream checkout: {folder}')
            subprocess.run(['git','-C',str(folder),'fetch','origin',commit],check=True)
            subprocess.run(['git','-C',str(folder),'checkout','--detach',commit],check=True)
        print(name,commit)
