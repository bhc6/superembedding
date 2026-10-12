"""Wait for the parent, run the independent control, then evaluate fixed exports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runner import OUT, PARENT, ROOT, digest, write_json

SCRIPTS = ['scripts/export_general_model.py', 'scripts/evaluate_validation_controls.py',
           'scripts/evaluate_mmeb_scienceqa.py', 'scripts/evaluate_general_transfer.py']


def paired_summary(a, b, id_key, groups=None):
    """Positive differences favor LoRA. Conditional on these fixed models."""
    import numpy as np
    aa = {p[id_key]: bool(p['correct']) for p in a}
    bb = {p[id_key]: bool(p['correct']) for p in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError('Prediction identity mismatch')
    groups = groups or {'all': sorted(aa)}
    flattened = [i for ids in groups.values() for i in ids]
    if len(flattened) != len(aa) or set(flattened) != aa.keys():
        raise ValueError('Incomplete or repeated comparison groups')
    rng = np.random.default_rng(20261011)
    boots, per_source = [], {}
    for source, ids in sorted(groups.items()):
        delta = np.array([float(aa[i])-float(bb[i]) for i in ids])
        boot = delta[rng.integers(0, len(ids), (5000, len(ids)))].mean(-1)
        boots.append(boot)
        per_source[source] = {'n': len(ids), 'delta': float(delta.mean()),
                              'paired_bootstrap_95': np.quantile(boot, [.025, .975]).tolist()}
    return {'lora_minus_frozen': sum(x['delta'] for x in per_source.values())/len(per_source),
            'paired_bootstrap_95': np.quantile(np.mean(boots, axis=0), [.025, .975]).tolist(),
            'per_group': per_source,
            'limitation': 'Query resampling, stratified macro average when grouped; not clustered by image/answer; not training-seed uncertainty or selection-adjusted.'}


def seal():
    protocol = {'protocol': 'v5-frozen-lora-final-evaluation-v1',
                'scripts': {f: digest(ROOT/f) for f in SCRIPTS},
                'pipeline_sha256': digest(__file__),
                'training_protocol_sha256': digest(OUT/'protocol.json'),
                'models': ['frozen_primary', 'frozen_validation_tuned', 'lora_validation_best'],
                'selection': 'Validation only. Frozen includes step 0; parent selects among positive validation steps.',
                'tasks': ['validation full and QA text-only', 'MMEB ScienceQA full and text-only',
                          'Urban1k common 999', 'evidence diagnostic'],
                'device': 'cuda', 'tuning_to_external_results': False,
                'scope': 'Retrospective single-paired-seed comparison; no publication/SOTA claim'}
    path = OUT/'evaluation_protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Evaluation implementation changed; do not silently reseal')
    write_json(protocol, path)


def run():
    from filelock import FileLock
    with FileLock(str(OUT/'pipeline.lock'), timeout=0):
        seal()
        state = json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
        if state.get('state') != 'completed':
            write_json({'state': 'waiting_or_training', 'pid': os.getpid()}, OUT/'pipeline_status.json')
            with (OUT/'queue.log').open('a', encoding='utf8') as log:
                subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).with_name('runner.py')), '--wait'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state = json.loads((OUT/'status.json').read_text())
        if state.get('state') != 'completed':
            write_json({'state': 'training_not_completed', 'detail': state}, OUT/'pipeline_status.json')
            return
        models = {'frozen_primary': OUT/'primary_model.pt',
                  'frozen_validation_tuned': OUT/'validation_tuned_model.pt',
                  'lora_validation_best': PARENT/'export/general_model.pt'}
        outputs = {}
        for label, model in models.items():
            if (OUT/'STOP').exists():
                write_json({'state': 'paused_before_evaluation', 'model': label}, OUT/'pipeline_status.json')
                return
            tag = 'general_'+digest(model)[:12]
            commands = []
            if label.startswith('frozen'):
                lr = '1e-05' if label == 'frozen_primary' else str(state['secondary_selected_lr'])
                commands.append([SCRIPTS[0], '--checkpoint', str(OUT/f'best_{lr}.pt'), '--output', str(model),
                                 '--device', 'cuda', '--image', str(ROOT/'data/evidence_diagnostic_v1/images/0000_anchor.png')])
            commands += [[SCRIPTS[1], '--backend', 'general', '--checkpoint', str(model), '--device', 'cuda'],
                         [SCRIPTS[2], '--backend', 'general', '--checkpoint', str(model), '--device', 'cuda'],
                         [SCRIPTS[2], '--backend', 'general', '--checkpoint', str(model), '--device', 'cuda',
                          '--query-mode', 'text_only'],
                         [SCRIPTS[3], '--checkpoint', str(model), '--device', 'cuda']]
            for i, command in enumerate(commands):
                seal()
                if (OUT/'STOP').exists():
                    write_json({'state': 'paused_before_evaluation', 'model': label, 'task': i}, OUT/'pipeline_status.json')
                    return
                write_json({'state': 'evaluating', 'model': label, 'task': i,
                            'command': command, 'pid': os.getpid()}, OUT/'pipeline_status.json')
                with (OUT/f'eval_{label}_{i}.log').open('a', encoding='utf8') as log:
                    subprocess.run([sys.executable, '-B', '-u', *command], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            result_root = ROOT/'artifacts/results/v5'
            paths = {'validation': result_root/'validation_controls'/f'{tag}_cuda.json',
                     'scienceqa_full': result_root/f'scienceqa_{tag}.json',
                     'scienceqa_text_only': result_root/f'scienceqa_{tag}_text_only.json',
                     'transfer': result_root/f'transfer_{tag}.json'}
            outputs[label] = {'checkpoint': str(model), 'sha256': digest(model),
                              'results': {k: str(p) for k,p in paths.items()},
                              'result_sha256': {k: digest(p) for k,p in paths.items()}}
        from superembedding.validation_protocol import load_protocol
        _, items, _ = load_protocol('D:/DEMO/superembedding-storage/superembedding/v5/data_v51')
        groups = {s: [r['id'] for r in items if r['source'] == s] for s in sorted({r['source'] for r in items})}
        results = {label: {k: json.loads(Path(p).read_text()) for k,p in v['results'].items()}
                   for label,v in outputs.items()}
        parent = results['lora_validation_best']
        comparisons = {}
        for label in ['frozen_primary', 'frozen_validation_tuned']:
            frozen = results[label]
            comparisons[label] = {
                'validation_macro': paired_summary(parent['validation']['full']['predictions'],
                                                    frozen['validation']['full']['predictions'], 'id', groups),
                'scienceqa': paired_summary(parent['scienceqa_full']['predictions'],
                                             frozen['scienceqa_full']['predictions'], 'query'),
                'urban1k': {k: parent['transfer']['urban1k_common_999'][k]-v
                            for k,v in frozen['transfer']['urban1k_common_999'].items()},
                'qa_image_gain': {name: results[name]['validation']['qa_full']['macro_r1']-
                                       results[name]['validation']['qa_text_only']['macro_r1']
                                  for name in ['lora_validation_best', label]}}
        write_json({'models': outputs, 'comparisons': comparisons,
                    'scope': 'Single paired training seed; validation-selected checkpoints; inspect per-source and transfer tradeoffs, not a universal LoRA superiority claim.'}, OUT/'final_comparison.json')
        write_json({'state': 'completed', 'result': str(OUT/'final_comparison.json'),
                    'completed_unix': time.time()}, OUT/'pipeline_status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.seal:
        seal()
    else:
        try:
            run()
        except Exception as exc:
            write_json({'state': 'failed', 'error': type(exc).__name__, 'message': str(exc),
                        'resume': 'Rerun pipeline.py after inspecting the failure; cached completed work is retained.'},
                       OUT/'pipeline_status.json')
            raise
