"""Atomic, resumable training snapshots. Load only trusted local checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source, destination):
    destination=Path(destination)
    temporary=destination.with_name(destination.name+'.tmp')
    with Path(source).open('rb') as src,temporary.open('wb') as dst:
        shutil.copyfileobj(src,dst)
        dst.flush();os.fsync(dst.fileno())
    os.replace(temporary,destination)


def atomic_json(value, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def save_training(path, model, optimizer, step, config, rng, extra=None, trainable_only=False):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
             if not trainable_only or k in names}
    atomic_save({'version': 1, 'model': state, 'partial': trainable_only,
                 'optimizer': optimizer.state_dict(), 'step': step,
                 'config': config, 'fingerprint': fingerprint(config),
                 'rng_local': rng.getstate(), 'rng_python': random.getstate(),
                 'rng_torch': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'extra': extra or {}}, path)


def restore_training(path, model, optimizer, config, rng):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['version'] != 1 or saved['fingerprint'] != fingerprint(config):
        raise ValueError('Resume configuration/data/model fingerprint mismatch')
    if saved['partial']:
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        if expected != set(saved['model']):
            raise ValueError('Trainable parameter names changed')
    model.load_state_dict(saved['model'], strict=not saved['partial'])
    optimizer.load_state_dict(saved['optimizer'])
    rng.setstate(saved['rng_local'])
    random.setstate(saved['rng_python'])
    torch.set_rng_state(saved['rng_torch'])
    if saved['rng_cuda']:
        if not torch.cuda.is_available():
            raise ValueError('This checkpoint requires CUDA to resume exactly')
        torch.cuda.set_rng_state_all(saved['rng_cuda'])
    return saved['step'], saved['extra']
