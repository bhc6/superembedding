"""Pinned, resumable D-drive downloads; no GPU use or remote-code execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json


def main():
    from huggingface_hub import snapshot_download
    storage=Path(os.environ.get('SUPEREMBEDDING_STORAGE','D:/DEMO/superembedding-storage'))
    output=storage/'official_models';output.mkdir(parents=True,exist_ok=True)
    if storage.resolve().drive.upper()!='D:':raise ValueError('Official checkpoints must be stored on D:')
    catalog=json.loads(Path('sources/official_embedding_models_v5.json').read_text())
    completed=[]
    for model in catalog['models']:
        repo=model['repo'];folder=output/repo.replace('/','--');revision=model['revision']
        if model.get('gated'):raise ValueError('Unexpected gated repository')
        if shutil.disk_usage(output).free<model['total_bytes']+10*2**30:raise RuntimeError('Insufficient D-drive reserve')
        atomic_json({'state':'downloading','repo':repo,'revision':revision,'completed':completed},output/'status.json')
        for attempt in range(3):
            try:
                snapshot_download(repo,revision=revision,local_dir=folder,
                    allow_patterns=[f['path'] for f in model['files']],max_workers=3)
                break
            except Exception as exc:
                atomic_json({'state':'retrying','repo':repo,'attempt':attempt+1,'error_type':type(exc).__name__,
                    'completed':completed},output/'status.json')
                if attempt==2:raise RuntimeError('Pinned download failed; partial files preserved') from None
                time.sleep(15)
        verified=[]
        for spec in model['files']:
            p=folder/spec['path']
            if not p.is_file() or (spec['bytes'] is not None and p.stat().st_size!=spec['bytes']):
                raise ValueError('Downloaded file size mismatch')
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if spec.get('lfs_sha256') and digest!=spec['lfs_sha256']:raise ValueError('Model weight checksum mismatch')
            verified.append({'file':spec['path'],'bytes':p.stat().st_size,'sha256':digest})
        info={'repo':repo,'revision':revision,'path':str(folder),'files':verified,
              'scope':'Official files downloaded and hashed only; inference reproduction pending; remote code not executed.'}
        atomic_json(info,folder/'download_verified.json');completed.append({k:v for k,v in info.items() if k!='files'})
        atomic_json({'state':'in_progress','models':completed},'artifacts/results/v5/official_model_downloads.json')
        print(json.dumps({'downloaded':repo,'revision':revision,'files':len(verified)}),flush=True)
    atomic_json({'state':'completed','models':completed},output/'status.json')
    atomic_json({'state':'completed','models':completed},'artifacts/results/v5/official_model_downloads.json')


if __name__=='__main__':
    from filelock import FileLock
    root=Path(os.environ.get('SUPEREMBEDDING_STORAGE','D:/DEMO/superembedding-storage'))/'official_models'
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'download.lock'),timeout=0):main()
