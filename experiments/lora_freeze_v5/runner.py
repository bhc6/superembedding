"""Independent, resumable frozen-backbone counterpart to the existing V5 run.

Never pauses, resumes or edits the parent experiment. GPU work waits for the
parent supervisor to finish its training AND final evaluation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')
OUT = STORE / 'lora_freeze_matched_v1'
PARENT = STORE / 'general_stage1_v51'
INITIAL = STORE / 'step0_reconstructed_cuda/checkpoint.pt'
LRS = (1e-5, 1e-4, 1e-3)
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def key_for(record, text):
    answer = ' '.join(text['assistant'].split())
    if record['task'] == 'caption':
        return ('image:' + record['original_rgb_sha256'], 'caption:' + answer.casefold())
    return answer.casefold().strip(' .!?')


def objects_for(record, text):
    answer = ' '.join(text['assistant'].split())
    if record['task'] == 'caption':
        return ({'text': 'Find an image matching this description: ' + answer},
                {'images': [record['image']]}, key_for(record, text))
    return ({'images': [record['image']],
             'text': 'Find the answer to this question: ' + text['user']},
            {'text': answer}, key_for(record, text))


def records():
    rows = [json.loads(s) for s in (STORE/'data_v51/records.jsonl').read_text(encoding='utf8').splitlines()]
    for row in rows:
        row['texts'] = [t for t in row['texts'] if t['assistant'].strip() and t['user'].strip()
                        and len(t['assistant']) < 5000]
    return [r for r in rows if r['texts']]


def schedule(rows, config):
    sources = sorted({r['source'] for r in rows if r['split'] == 'train'})
    pools = {s: [r for r in rows if r['split'] == 'train' and r['source'] == s] for s in sources}
    rng = random.Random(config['seed'])
    queue = []
    for step in range(1, config['steps'] + 1):
        if not queue:
            queue = sources.copy()
            rng.shuffle(queue)
        source = queue.pop()
        pool = [r for r in pools[source] if step > 500 or r['order'] < 256]
        for _ in range(32):
            batch = rng.sample(pool, config['batch_size'])
            indices = [rng.randrange(len(r['texts'])) for r in batch]
            keys = [key_for(r, r['texts'][i]) for r, i in zip(batch, indices)]
            groups = [set(k) if isinstance(k, tuple) else {k} for k in keys]
            if not all(a & b for a in groups for b in groups):
                break
        else:
            raise RuntimeError('No negatives after 32 samples')
        yield {'step': step, 'source': source,
               'samples': [[r['id'], i] for r, i in zip(batch, indices)]}, rng.getstate(), queue.copy()


def lr_scale(step, total):
    warmup = min(100, max(1, total // 10))
    return (step / warmup if step <= warmup else
            .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def prepare():
    """Seal schedule and compare local RNG against the preserved step-60 state."""
    os.chdir(ROOT)
    import torch
    from scripts.train_general import pair
    config = json.loads((PARENT/'config.json').read_text())
    assert digest(STORE/'data_v51/records.jsonl') == config['data_sha256']
    for file, sha in config['implementation_sha256'].items():
        assert digest(ROOT/file) == sha, ('Historical implementation changed', file)
    rows = records()
    lookup = {r['id']: r for r in rows}
    parent = torch.load(STORE/'general_stage1/last.pt', map_location='cpu', weights_only=True)
    assert parent['step'] == 60
    history = {}
    for folder in [STORE/'general_stage1', PARENT]:
        for line in (folder/'history.jsonl').read_text().splitlines():
            item = json.loads(line)
            history[item['step']] = item['source']
    plan = []
    source_checks = 0
    for item, rng, queue in schedule(rows, config):
        if item['step'] in history:
            assert history[item['step']] == item['source']
            source_checks += 1
        if item['step'] == 60:
            assert rng == parent['rng_local'], 'Sampling RNG differs at step 60'
            assert queue == parent['extra']['source_queue'], 'Source queue differs at step 60'
        plan.append(item)
    # Compare each type of object construction to the actual training function.
    for row in rows:
        rr = random.Random(row['id'])
        reference = pair(row, rr)
        text = random.Random(row['id']).choice(row['texts'])
        assert reference == objects_for(row, text)
    initial = torch.load(INITIAL, map_location='cpu', weights_only=True)
    assert initial['step'] == 0 and initial['config'] == config
    assert initial['reconstruction']['cpu_rng_matches_parent']
    identity = {'protocol': 'v5-frozen-lora-matched-v1', 'parent_config': config,
                'schedule_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                'initial_checkpoint_sha256': digest(INITIAL),
                'runner_sha256': digest(__file__), 'learning_rates': list(LRS),
                'primary_lr': 1e-5, 'seed': config['seed'],
                'precision': 'CUDA BF16 backbone; FP32 shared head, normalization and scores',
                'initialization_limitation': 'Reconstructed step 0; original file/hash unavailable',
                'parent_constructor_rng_check': True, 'sampling_rng_step60_exact': True,
                'sampling_source_queue_step60_exact': True,
                'scope': 'One paired seed; matched updates/data, not matched wall-clock/FLOPs',
                'selection': 'Primary 1e-5. Secondary LR chosen only on validation; ties use lower LR/earlier step.',
                'validation_steps': [0] + list(range(250, 2001, 250))}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'protocol.json'
    if target.exists():
        if json.loads(target.read_text()) != identity or json.loads((OUT/'schedule.json').read_text()) != plan:
            raise ValueError('Existing sealed experiment differs; use a new experiment directory')
    else:
        write_json(plan, OUT/'schedule.json')
        write_json(identity, target)
    write_json({'matched_logged_source_steps': source_checks, 'planned_updates': len(plan),
                'rng_at_step60_exact': True, 'source_queue_at_step60_exact': True,
                'object_construction_matches_reference': True,
                'train_records': sum(r['split'] == 'train' for r in rows),
                'validation_records': sum(r['split'] == 'val' for r in rows)}, OUT/'preflight.json')
    print(json.dumps(json.loads((OUT/'preflight.json').read_text())), flush=True)


def wait_and_run():
    from filelock import FileLock
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT/'queue.lock'), timeout=0):
        deadline = time.monotonic() + 96 * 3600
        stable = 0
        while time.monotonic() < deadline:
            if (OUT/'STOP').exists():
                write_json({'state': 'stopped_before_training'}, OUT/'status.json')
                return
            state = json.loads((PARENT/'supervisor.json').read_text()).get('state')
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'], text=True).splitlines()[0])
            ready = state == 'stage_complete' and free >= 4500
            stable = stable + 1 if ready else 0
            write_json({'state': 'waiting_for_parent_training_and_final_evaluation',
                        'parent_supervisor_state': state, 'free_gpu_MiB': free,
                        'required_free_gpu_MiB': 4500, 'pid': os.getpid(),
                        'primary_lr': 1e-5, 'updates_planned': 2000,
                        'condition': 'parent supervisor stage_complete; >=4500 MiB for three polls',
                        'updated_unix': time.time()}, OUT/'status.json')
            if stable >= 3:
                with (OUT/'worker.log').open('a', encoding='utf8') as log:
                    result = subprocess.run([sys.executable, '-B', '-u', __file__, '--train'],
                                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write_json({'state': 'failed', 'exit_code': result.returncode,
                                'log': str(OUT/'worker.log'), 'resume': 'same --wait command'}, OUT/'status.json')
                return
            time.sleep(30)
        write_json({'state': 'wait_timeout', 'hours': 96,
                    'action': 'Inspect parent state; rerun --wait to continue'}, OUT/'status.json')


def train():
    os.chdir(ROOT)
    os.environ.setdefault('MSO_FLA', '0')
    os.environ.setdefault('MSO_BRANCH_CHUNK', '4')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from filelock import FileLock
    from superembedding.checkpoint import atomic_save, fingerprint
    from superembedding.general import cached_backward, positive_mask
    from superembedding.general_export import load_bundle_encoder, export_checkpoint
    from superembedding.heads import OptionEmbeddingHead
    with FileLock(str(OUT/'training.lock'), timeout=0):
        protocol = json.loads((OUT/'protocol.json').read_text())
        config = protocol['parent_config']
        if digest(__file__) != protocol['runner_sha256'] or digest(INITIAL) != protocol['initial_checkpoint_sha256']:
            raise ValueError('Sealed implementation/initialization changed')
        if digest(STORE/'data_v51/records.jsonl') != config['data_sha256']:
            raise ValueError('Training data changed')
        if not torch.cuda.is_available():
            raise RuntimeError('Matched experiment requires CUDA')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config['seed'])
        initial = torch.load(INITIAL, map_location='cpu', weights_only=True)
        model, _ = load_bundle_encoder(STORE/'step0_reconstructed_cuda/general_model.pt')
        model.eval()
        current = model.state_dict()
        if any(not torch.equal(current[k].detach().cpu(), v) for k,v in initial['model'].items()):
            raise ValueError('Loaded initial bundle differs from sealed initial checkpoint')
        del current
        assert not any(p.requires_grad for p in model.backbone.parameters())
        device = next(model.backbone.parameters()).device
        assert device.type == 'cuda'
        # Save all original LoRA tensors in every export; frozen still means present.
        original_lora = {k: v for k, v in initial['model'].items() if k.startswith('backbone.')}
        heads = {str(lr): deepcopy(model.embedding_head).to(device).eval() for lr in LRS}
        optimizers = {k: torch.optim.AdamW(h.parameters(), lr=float(k), weight_decay=.01) for k, h in heads.items()}
        rows = records()
        lookup = {r['id']: r for r in rows}
        image_hashes = {r['image']: r['image_file_sha256'] for r in rows}
        checked_images = set()
        val = {s: sorted([r for r in rows if r['split'] == 'val' and r['source'] == s], key=lambda r:r['id'])
               for s in sorted({r['source'] for r in rows if r['split'] == 'train'})}
        plan = json.loads((OUT/'schedule.json').read_text())
        if hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() != protocol['schedule_sha256']:
            raise ValueError('Sealed sample schedule changed')
        cache = OUT/'features'
        cache.mkdir(exist_ok=True)
        cache_identity = fingerprint({'initial': protocol['initial_checkpoint_sha256'],
                                      'encoder': config['implementation_sha256'],
                                      'dtype': 'cuda-bfloat16', 'pixels': config['max_pixels']})
        feature_count = 0

        def feature(obj):
            nonlocal feature_count
            k = fingerprint({'object': obj, 'identity': cache_identity})
            path = cache/(k + '.pt')
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                u = saved['u']
                if saved['key'] != k or u.shape != (4, 1024) or not torch.isfinite(u).all():
                    raise ValueError('Corrupt cached feature')
            else:
                for image in obj.get('images', []):
                    if image not in checked_images:
                        if digest(image) != image_hashes[image]:
                            raise ValueError('Source image changed')
                        checked_images.add(image)
                with torch.no_grad():
                    u, _ = model.extract_features(obj)
                u = u.detach().cpu()
                if u.shape != (4, 1024) or not torch.isfinite(u).all():
                    raise ValueError('Invalid feature')
                atomic_save({'key': k, 'u': u}, path)
                feature_count += 1
            return u.to(device)

        def evaluate():
            scores = {k: {} for k in heads}
            with torch.no_grad():
                for source, records_ in val.items():
                    pairs = [objects_for(r, random.Random(r['id']).choice(r['texts'])) for r in records_]
                    uq = [feature(p[0]) for p in pairs]
                    ud = [feature(p[1]) for p in pairs]
                    mask = positive_mask([p[2] for p in pairs], 'cpu')
                    for k, head in heads.items():
                        q = torch.stack([head(u).embedding for u in uq])
                        d = torch.stack([head(u).embedding for u in ud])
                        pred = (q@d.T).argmax(-1).cpu()
                        scores[k][source] = mask[torch.arange(len(pairs)), pred].float().mean().item()
                    print(json.dumps({'validation_source': source, 'new_features': feature_count}), flush=True)
            return {k: {'macro_r1': sum(v.values())/len(v), 'per_source_r1': v} for k,v in scores.items()}

        best = {k: {'macro_r1': -1., 'step': 0} for k in heads}
        start = 0
        history = []
        protocol_hash = fingerprint(protocol)
        if (OUT/'last.pt').exists():
            saved = torch.load(OUT/'last.pt', map_location='cpu', weights_only=True)
            if saved['protocol_hash'] != protocol_hash:
                raise ValueError('Resume fingerprint differs')
            for k,h in heads.items():
                h.load_state_dict(saved['heads'][k], strict=True)
                optimizers[k].load_state_dict(saved['optimizers'][k])
            start, best, history = saved['step'], saved['best'], saved['history']
            torch.set_rng_state(saved['rng_torch'])
            torch.cuda.set_rng_state_all(saved['rng_cuda'])

        def save(step):
            atomic_save({'protocol_hash': protocol_hash, 'step': step,
                         'heads': {k: {n:p.detach().cpu() for n,p in h.state_dict().items()} for k,h in heads.items()},
                         'optimizers': {k:o.state_dict() for k,o in optimizers.items()},
                         'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
                         'history': history, 'best': best, 'next_schedule_index': step}, OUT/'last.pt')

        def record_validation(step):
            metrics = evaluate()
            for k, v in metrics.items():
                if v['macro_r1'] > best[k]['macro_r1']:
                    best[k] = {'macro_r1': v['macro_r1'], 'step': step}
                    state = {**original_lora, **{'embedding_head.'+n:p.detach().cpu().clone()
                                               for n,p in heads[k].state_dict().items()}}
                    arm_config = deepcopy(config)
                    arm_config.update(lr=float(k), protocol='v5-frozen-lora-matched-v1',
                                      treatment='frozen original OmniJev; projection only',
                                      matched_parent_protocol_hash=protocol_hash,
                                      initialization='reconstructed CUDA step0')
                    atomic_save({'model': state, 'step': step, 'config': arm_config,
                                 'fingerprint': fingerprint(arm_config),
                                 'experiment': {'protocol_hash': protocol_hash, 'frozen_backbone': True,
                                                'head_lr': float(k), 'note': 'config describes inherited encoder; training treatment here'}},
                                OUT/f'best_{k}.pt')
            history.append({'step': step, 'validation': metrics})
            write_json(history, OUT/'history.json')
            compare = {}
            for line in (PARENT/'history.jsonl').read_text().splitlines():
                entry = json.loads(line)
                if 'validation' in entry:
                    compare[str(entry['step'])] = entry['validation']
            write_json({'state': 'partial', 'frozen': history, 'lora_parent': compare,
                        'primary_lr': 1e-5, 'limitation': protocol['scope']}, OUT/'comparison.json')
            print(json.dumps({'step': step, 'validation': metrics}), flush=True)

        if not (OUT/'last.pt').exists():
            record_validation(0)
            # Initial validation must reproduce the independently reconstructed CUDA baseline.
            ref = json.loads((ROOT/'artifacts/results/v5/validation_controls/step0_reconstructed_cuda.json').read_text())['full']
            measured = history[-1]['validation']['1e-05']
            if any(abs(measured['per_source_r1'][s]-v['R1']) > 1e-6 for s,v in ref['per_source'].items()):
                raise ValueError('Step-0 validation does not reproduce matched CUDA baseline')
            save(0)
        if (OUT/'STOP').exists():
            write_json({'state': 'paused', 'step': start}, OUT/'status.json')
            return
        for item in plan[start:]:
            step = item['step']
            pairs = [objects_for(lookup[rid], lookup[rid]['texts'][i]) for rid,i in item['samples']]
            uq = [feature(p[0]) for p in pairs]
            ud = [feature(p[1]) for p in pairs]
            keys = [p[2] for p in pairs]
            entries = {}
            for k, head in heads.items():
                optimizer = optimizers[k]
                optimizer.zero_grad(set_to_none=True)
                loss, error = cached_backward(lambda u: head(u).embedding, uq, ud, keys)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite head gradient')
                for group in optimizer.param_groups:
                    group['lr'] = float(k)*lr_scale(step, config['steps'])
                optimizer.step()
                entries[k] = {'loss': loss, 'gradient_replay_max_error': error}
            if step % 250 == 0:
                record_validation(step)
            if step % 5 == 0 or (OUT/'STOP').exists():
                save(step)
            if step % 500 == 0:
                for k, head in heads.items():
                    atomic_save({'head': {n:p.detach().cpu() for n,p in head.state_dict().items()},
                                 'step': step, 'protocol_hash': protocol_hash}, OUT/f'step_{step:06d}_{k}_head.pt')
            write_json({'state': 'running', 'step': step, 'source': item['source'],
                        'arms': entries, 'new_features_this_launch': feature_count,
                        'pid': os.getpid()}, OUT/'status.json')
            if (OUT/'STOP').exists():
                write_json({'state': 'paused', 'step': step}, OUT/'status.json')
                return
        selected = max(heads, key=lambda k: (best[k]['macro_r1'], -float(k)))
        # Two explicit exports: primary controlled LR and secondary validation-tuned LR.
        for label,k in [('primary', '1e-05'), ('validation_tuned', selected)]:
            export_checkpoint(OUT/f'best_{k}.pt', OUT/f'{label}_model.pt')
        write_json({'state': 'completed', 'steps': config['steps'], 'best': best,
                    'secondary_selected_lr': float(selected), 'primary_lr': 1e-5,
                    'external_evaluation': 'pending; no external results used for selection',
                    'decision_retention': 'not measured', 'scope': protocol['scope']}, OUT/'status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--wait', action='store_true')
    group.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.wait:
        wait_and_run()
    else:
        train()
