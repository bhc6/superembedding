"""Exact continuation of an audit-paused run; old validation is diagnostic only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_copy, atomic_json
from superembedding.leakage_guard import digest

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')
PARENT = STORE/'general_stage1_v51'
OUT = STORE/'general_step556_continuation_v1'
DATA = STORE/'data_v51'


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify(protocol):
    for path, sha in protocol['sealed_files'].items():
        if digest(path) != sha:
            raise ValueError('Continuation input changed: ' + path)
    if digest(OUT/'initial_step556.pt') != protocol['parent_checkpoint_sha256']:
        raise ValueError('Initial checkpoint changed')
    # The historical paused run stays untouched, including its STOP marker.
    if digest(PARENT/'last.pt') != protocol['parent_checkpoint_sha256']:
        raise ValueError('Historical parent checkpoint changed')


def prepare():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'continuation_protocol.json').exists():
        protocol = json.loads((OUT/'continuation_protocol.json').read_text())
        verify(protocol)
        return protocol
    if any(OUT.iterdir()):
        raise ValueError('Unsealed partial preparation: inspect before continuing')
    saved = torch.load(PARENT/'last.pt', map_location='cpu', weights_only=True)
    config = json.loads((PARENT/'config.json').read_text())
    if saved['step'] != 556 or saved['config'] != config or config['steps'] != 2000:
        raise ValueError('Unexpected parent checkpoint/configuration')
    if len(saved['optimizer']['state']) != 304 or not all(k in saved for k in ['rng_local','rng_python','rng_torch','rng_cuda']):
        raise ValueError('Incomplete optimizer/RNG state')
    if digest(DATA/'records.jsonl') != config['data_sha256']:
        raise ValueError('Parent data changed')
    sealed = {str(ROOT/p):sha for p,sha in config['implementation_sha256'].items()}
    sealed[str(DATA/'records.jsonl')] = config['data_sha256']
    sealed[str(Path(__file__).resolve())] = digest(__file__)
    for path, sha in sealed.items():
        if digest(path) != sha:
            raise ValueError('Historical implementation changed: ' + path)
    for name in ['config.json','last.pt','best.pt']:
        atomic_copy(PARENT/name,OUT/name)
    atomic_copy(PARENT/'last.pt',OUT/'initial_step556.pt')
    atomic_copy(PARENT/'history.jsonl',OUT/'parent_history_61_556.jsonl')
    protocol = {
        'protocol':'step556-exact-diagnostic-continuation-v1', 'sealed_utc':utc(),
        'parent_run':str(PARENT), 'parent_step':556,
        'parent_checkpoint_sha256':digest(PARENT/'last.pt'), 'sealed_files':sealed,
        'training_data_unchanged':True, 'resume_optimizer_rng_and_sampler':True,
        'terminal_step':2000, 'terminal_rule':'Original pre-audit fixed 2000 updates; never select by observed scores.',
        'old_validation_use':'Diagnostic only. Trainer still records best.pt for historical numerical equivalence; it is NOT the confirmatory checkpoint.',
        'confirmatory_checkpoint':'Fixed terminal step 2000, subject to separate evaluation clearance. No automatic external evaluation.',
        'new_holdout_boundary':'All historical train AND validation manifests, including pre-v51 data, are development exposure. New holdouts require image/document/entity/scene and text-overlap audit against that union.',
        'not_a_clean_retraining_claim':True, 'release_eligible':False,
        'scope':'Continue useful V5 learned weights; does not repair CUB/DTD historical experiments or certify no pretraining contamination.'}
    atomic_json(protocol,OUT/'continuation_protocol.json')
    atomic_json(protocol,ROOT/'artifacts/results/v5/step556_continuation_protocol.json')
    verify(protocol)
    return protocol


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='Resume exactly one update to step 557 and save full state')
    args=ap.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        p=prepare(); print(json.dumps({'prepared':str(OUT),'parent_step':p['parent_step']})); return
    from filelock import FileLock
    protocol=json.loads((OUT/'continuation_protocol.json').read_text())
    with FileLock(str(OUT/'continuation.lock'),timeout=0):
        verify(protocol)
        if (OUT/'STOP_AFTER_CHECKPOINT').exists():
            raise RuntimeError('Continuation pause marker exists')
        command=[sys.executable,'-u','scripts/train_general.py','--data',str(DATA),'--output',str(OUT),
                 '--steps','2000','--batch-size','16','--checkpoint-every','5','--validate-every','250','--milestone-every','500']
        if args.smoke: command+=['--stop-after','557']
        atomic_json({'state':'running','pid':os.getpid(),'command':command,'utc':utc(),
                     'formal_selection_enabled':False},OUT/'continuation_status.json')
        with (OUT/'continuation_worker.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status=json.loads((OUT/'status.json').read_text())
        verify(protocol)
        if result.returncode:
            atomic_json({'state':'failed','exit_code':result.returncode,'training_status':status,
                         'resume':'Rerun same wrapper after inspecting failure; no automatic unbounded retries'},OUT/'continuation_status.json')
            raise RuntimeError('Continuation failed; checkpoint preserved')
        # Historical trainer exports a head-only object: quarantine it immediately.
        legacy=OUT/'embedding_head.pt'
        if legacy.exists():
            target=OUT/'terminal_head_only_DO_NOT_DEPLOY.pt'
            if target.exists():raise ValueError('Head quarantine destination exists')
            legacy.replace(target)
        state='completed_diagnostic_training' if status['state']=='completed' else status['state']
        atomic_json({'state':state,'training_status':status,'utc':utc(),
                     'release_eligible':False,'external_evaluation_started':False,
                     'next':'Audit fresh isolated holdouts; use fixed terminal checkpoint, not old validation best.'},OUT/'continuation_status.json')


if __name__=='__main__':
    main()
