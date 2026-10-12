"""New queue for sealed frozen control, gated on exact continuation completion.

Historical runner and STOP are preserved. All validation-selected outputs are
diagnostic. This wrapper also reconstructs fixed-terminal complete checkpoints.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superembedding.checkpoint import atomic_json, atomic_save, fingerprint
from superembedding.leakage_guard import digest
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')
OUT = STORE/'frozen_terminal_control_v2'
CONTINUATION = STORE/'general_step556_continuation_v1'
RUNNER = ROOT/'experiments/lora_freeze_v5/runner.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('sealed_frozen_runner', RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    return module


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'protocol': 'frozen-terminal-control-queue-v2', 'runner_sha256': digest(RUNNER),
            'wrapper_sha256': digest(__file__),
            'continuation_protocol_sha256': digest(CONTINUATION/'continuation_protocol.json'),
            'frozen_schedule': 'Original sealed 2000-step schedule, reconstructed Step0, three predefined LRs',
            'terminal_rule': 'Fixed step2000; primary LR1e-5. Other LRs and validation-best are diagnostic.',
            'gpu_gate': 'Continuation completed_diagnostic_training plus >=4500MiB free at three consecutive polls',
            'scope': 'Retrospective paired-seed control on historical development data; no fresh holdout scoring or release',
            'release_eligible': False}
    seal = OUT/'queue_protocol.json'
    if seal.exists() and json.loads(seal.read_text()) != plan:
        raise ValueError('Queue seal changed')
    atomic_json(plan, seal)
    load_runner().prepare()
    return plan


def train():
    import torch
    from superembedding.general_export import export_checkpoint
    plan = prepare()
    runner = load_runner()
    runner.train()
    state = json.loads((OUT/'status.json').read_text())
    if state.get('state') != 'completed':
        return state
    saved = torch.load(OUT/'last.pt', map_location='cpu', weights_only=True)
    if saved['step'] != 2000:
        raise ValueError('Fixed terminal checkpoint is incomplete')
    initial = torch.load(runner.INITIAL, map_location='cpu', weights_only=True)
    for lr, head in saved['heads'].items():
        config = deepcopy(initial['config'])
        config.update(lr=float(lr), treatment='frozen original OmniJev; projection only',
                      terminal_control_protocol_sha256=digest(OUT/'queue_protocol.json'),
                      checkpoint_rule='fixed terminal Step2000; not validation selected')
        tensors = {k: v for k, v in initial['model'].items() if k.startswith('backbone.')}
        tensors.update({'embedding_head.'+k: v for k, v in head.items()})
        checkpoint = OUT/f'terminal_{lr}.pt'
        if not checkpoint.exists():
            atomic_save({'step': 2000, 'config': config, 'fingerprint': fingerprint(config),
                         'model': tensors, 'experiment': plan}, checkpoint)
        export_checkpoint(checkpoint, OUT/f'terminal_{lr}_model.pt')
    return {'state': 'completed_diagnostic_control', 'fixed_step': 2000,
            'primary_lr': 1e-5, 'roundtrip_inference_audit': 'pending', 'release_eligible': False}


def main():
    from filelock import FileLock
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--train', action='store_true')
    args = ap.parse_args()
    plan = prepare()
    if args.prepare:
        print(json.dumps(plan)); return
    if args.train:
        atomic_json(train(), OUT/'queue_status.json'); return
    with FileLock(str(OUT/'queue.lock'), timeout=0):
        stable = 0
        while True:
            if (OUT/'STOP').exists():
                atomic_json({'state': 'paused_at_queue_boundary'}, OUT/'queue_status.json'); return
            parent = json.loads((CONTINUATION/'continuation_status.json').read_text())
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'], text=True).splitlines()[0])
            ready = parent.get('state') == 'completed_diagnostic_training' and free >= 4500
            stable = stable+1 if ready else 0
            atomic_json({'state': 'waiting_for_continuation', 'parent_state': parent.get('state'),
                         'free_gpu_MiB': free, 'pid': os.getpid(), 'stable_polls': stable,
                         'updates_planned': 2000, 'lrs': [1e-5, 1e-4, 1e-3]}, OUT/'queue_status.json')
            if stable >= 3:
                with (OUT/'worker.log').open('a', encoding='utf-8') as log:
                    code = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--train'],
                                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    atomic_json({'state': 'failed', 'exit_code': code, 'log': str(OUT/'worker.log'),
                                 'resume': 'Inspect and rerun; optimizer checkpoints retained'}, OUT/'queue_status.json')
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
