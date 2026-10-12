"""Restartable official-model inference on already-seen diagnostic benchmarks.

No training, selection, publication or use of fresh confirmatory test data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORE = Path('D:/DEMO/superembedding-storage')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['wemm2b', 'gme2b'], required=True)
    ap.add_argument('--max-new', type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from superembedding.checkpoint import atomic_json, atomic_save, fingerprint
    from superembedding.validation_protocol import load_protocol, score_vectors, paired_delta
    from filelock import FileLock
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError('CPU-only worker expected')
    name = {'wemm2b': 'tencent--WeMM-Embedding-2B',
            'gme2b': 'Alibaba-NLP--gme-Qwen2-VL-2B-Instruct'}[a.model]
    folder = STORE/'official_models'/name
    verified = read(folder/'download_verified.json')
    # Bind all model bytes, not just the mutable local folder name.
    for item in verified['files']:
        if sha(folder/item['file']) != item['sha256']:
            raise ValueError('Official model file changed: '+item['file'])
    protocol, items, _ = load_protocol(STORE/'superembedding/v5/data_v51')
    implementation = ['scripts/evaluate_paper_baseline_cpu.py', 'superembedding/validation_protocol.py',
                      'scripts/train_general.py', 'superembedding/general.py']
    import pyarrow.parquet as pq
    science = STORE/'superembedding/v5/mmeb_scienceqa'
    table = science/'test.parquet'
    science_rows = pq.read_table(table).to_pylist()
    if len(science_rows) != 1000:
        raise ValueError('Incomplete ScienceQA protocol')
    science_inputs = {'parquet_sha256': sha(table),
                      'image_sha256': {r['qry_img_path']: sha(science/r['qry_img_path']) for r in science_rows}}
    prov = {'protocol': 'paper-baseline-cpu-v3', 'model': a.model, 'download': verified,
            'scienceqa_inputs': science_inputs,
            'validation_protocol': protocol, 'device': 'cpu', 'dtype': 'float32', 'threads': 2,
            'max_pixels': 65536, 'max_text_tokens': 1800,
            'implementation': {p: sha(ROOT/p) for p in implementation},
            'torch': str(torch.__version__), 'transformers': __import__('transformers').__version__,
            'selection': 'No fitting or model/prompt selection; evaluate fixed official weights.',
            'scope': 'Seen diagnostic validation and ScienceQA. Restricted image budget; not paper headline reproduction.',
            'pretraining_contamination': 'Unknown; benchmark may occur in upstream embedding training.'}
    out = STORE/'superembedding/v5/paper_baselines_cpu_v3'/a.model
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/'worker.lock'), timeout=0):
        if (out/'protocol.json').exists() and read(out/'protocol.json') != prov:
            raise ValueError('Sealed baseline protocol changed')
        atomic_json(prov, out/'protocol.json')
        signature = fingerprint(prov)
        model = processor = None
        loading_info = None
        new = 0
        started = time.perf_counter()

        def setup():
            nonlocal model, processor, loading_info
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                                      min_pixels=3136, max_pixels=65536)
            processor.tokenizer.padding_side = 'right'
            if a.model == 'wemm2b':
                # Reviewed local source: native Qwen3.5 forward, reset rope_deltas,
                # last non-pad pooling and L2 normalization. No remote-code loader.
                source = folder/'modeling_wemm_embedding.py'
                spec = importlib.util.spec_from_file_location('reviewed_wemm_cpu', source)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                model, loading_info = module.WeMMEmbedding.from_pretrained(
                    folder, dtype=torch.float32, device_map='cpu', attn_implementation='sdpa',
                    local_files_only=True, output_loading_info=True)
            else:
                from transformers import Qwen2VLForConditionalGeneration
                model, loading_info = Qwen2VLForConditionalGeneration.from_pretrained(
                    folder, dtype=torch.float32, device_map='cpu', attn_implementation='sdpa',
                    local_files_only=True, output_loading_info=True)
            bad = {k: loading_info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']
                   if loading_info.get(k)}
            if bad:
                atomic_json(bad, out/'loading_failure.json')
                raise ValueError('Model weights were not loaded completely')
            model.eval()
            # Transformers 5 includes sets in auxiliary loading metadata.
            serial = json.loads(json.dumps(loading_info, default=lambda v: sorted(v) if isinstance(v, set) else str(v)))
            atomic_json({'passed': True, 'loading_info': serial}, out/'weight_load_audit.json')

        @torch.inference_mode()
        def encode(obj):
            if model is None:
                setup()
            paths = obj.get('images', [])
            if len(paths) > 1:
                raise ValueError('This protocol uses at most one image')
            images = [Image.open(p).convert('RGB') for p in paths]
            if a.model == 'wemm2b':
                content = ([{'type': 'image', 'image': im} for im in images] +
                           ([{'type': 'text', 'text': obj['text']}] if obj.get('text') else []))
                text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                     tokenize=False, add_generation_prompt=False)
            else:
                prefix = '<|vision_start|><|image_pad|><|vision_end|>' if images else ''
                text = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                        '<|im_start|>user\n'+prefix+obj.get('text', '')+
                        '<|im_end|>\n<|im_start|>assistant\n<|endoftext|>')
            inputs = processor(text=[text], images=images or None, return_tensors='pt',
                               padding=True, truncation=True, max_length=1800)
            if images and ('pixel_values' not in inputs or not inputs['pixel_values'].numel()):
                raise ValueError('Vision input silently disappeared')
            if a.model == 'wemm2b':
                end = processor.tokenizer.convert_tokens_to_ids('<embedding>')
                if int(inputs['input_ids'][0, -1]) != end:
                    raise ValueError('Official tokenizer must append the embedding token')
                v = model.embedding(**inputs, use_cache=False)[0]
            else:
                # GME's published forward uses inputs_embeds with no image-grid
                # position IDs: all three RoPE axes are ordinary token positions.
                # Preserve that behavior explicitly with Transformers 5's layout.
                n = inputs['input_ids'].shape[1]
                inputs['position_ids'] = torch.arange(n).view(1, 1, n).expand(3, 1, n)
                model.model.rope_deltas = None
                hidden = model.model(**inputs, use_cache=False).last_hidden_state
                v = F.normalize(hidden[0, int(inputs['attention_mask'][0].sum())-1].float(), dim=-1)
            if v.ndim != 1 or not torch.isfinite(v).all() or abs(float(v.norm())-1) > 1e-4:
                raise ValueError('Invalid embedding')
            return v.float().cpu()

        def key(obj):
            return fingerprint({'object': obj, 'image_sha256': [sha(p) for p in obj.get('images', [])],
                                'protocol': signature})

        def vector(obj, task):
            nonlocal new
            k = key(obj)
            path = out/'vectors'/(k+'.pt')
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['fingerprint'] != k or not torch.isfinite(saved['vector']).all():
                    raise ValueError('Cache identity/integrity mismatch')
                return saved['vector']
            if (out/'STOP_AFTER_VECTOR').exists() or (a.max_new and new >= a.max_new):
                raise InterruptedError('Paused at vector boundary')
            v = encode(obj)
            atomic_save({'fingerprint': k, 'vector': v}, path)
            new += 1
            atomic_json({'state': 'running', 'task': task, 'new_vectors': new, 'pid': os.getpid(),
                         'elapsed_seconds': time.perf_counter()-started,
                         'updated_utc': datetime.now(timezone.utc).isoformat()}, out/'status.json')
            print(json.dumps({'model': a.model, 'task': task, 'new_vectors': new}), flush=True)
            return v

        try:
            # Requested VSR first; same 110 queries and candidate bank as prior arms.
            vsr = [r for r in items if r['source'] == 'vsr']
            for subset, label in [(vsr, 'vsr110'), (items, 'validation760')]:
                result_path = out/(label+'.json')
                if result_path.exists():
                    if read(result_path)['protocol_fingerprint'] != signature:
                        raise ValueError('Result protocol mismatch')
                    continue
                qa = [r for r in subset if r['task'] == 'qa']
                q = torch.stack([vector(r['q'], label) for r in subset])
                d = torch.stack([vector(r['d'], label) for r in subset])
                t = torch.stack([vector({'text': r['q']['text']}, label+'_text_only') for r in qa])
                ix = [i for i, r in enumerate(subset) if r['task'] == 'qa']
                full = score_vectors(subset, q, d)
                text_only = score_vectors(qa, t, d[ix])
                qa_full = score_vectors(qa, q[ix], d[ix])
                result = {'protocol_fingerprint': signature, 'provenance': prov, 'full': full,
                          'qa_full': qa_full, 'qa_text_only': text_only,
                          'paired_image_gain': paired_delta(qa, qa_full, text_only)}
                atomic_json(result, result_path)
                atomic_json(result, ROOT/'artifacts/results/v5/paper_baselines_cpu_v3'/a.model/(label+'.json'))
            data = science_rows
            dataset_sha = science_inputs['parquet_sha256']
            for mode in ['full', 'text_only']:
                preds = []
                for i, row in enumerate(data):
                    obj = {'text': row['qry_text'].replace('<|image_1|>', '')}
                    if mode == 'full':
                        obj['images'] = [str(science/row['qry_img_path'])]
                    q = vector(obj, 'scienceqa_'+mode)
                    dv = torch.stack([vector({'text': s}, 'scienceqa_targets') for s in row['tgt_text']])
                    rank = int((q@dv.T).argmax())
                    preds.append({'query': i, 'candidate_index': rank,
                                  'correct': row['tgt_text'][rank] == row['tgt_text'][0]})
                result = {'protocol_fingerprint': signature, 'dataset_sha256': dataset_sha,
                          'metric': 'Original per-query candidate list, first target positive; exact text equality',
                          'provenance': prov, 'mode': mode, 'n': len(preds),
                          'accuracy': sum(p['correct'] for p in preds)/len(preds), 'predictions': preds}
                atomic_json(result, out/('scienceqa_'+mode+'.json'))
                atomic_json(result, ROOT/'artifacts/results/v5/paper_baselines_cpu_v3'/a.model/('scienceqa_'+mode+'.json'))
            if any(sha(ROOT/p) != h for p, h in prov['implementation'].items()):
                raise ValueError('Implementation changed during inference')
            if sha(table) != dataset_sha or any(sha(science/p) != h for p, h in science_inputs['image_sha256'].items()):
                raise ValueError('ScienceQA inputs changed during inference')
            atomic_json({'state': 'completed', 'pid': os.getpid(), 'new_vectors': new,
                         'scope': prov['scope']}, out/'status.json')
        except InterruptedError:
            atomic_json({'state': 'paused', 'new_vectors': new, 'resume': 'Same command; per-vector cache retained'}, out/'status.json')
        except Exception as exc:
            atomic_json({'state': 'failed', 'error_type': type(exc).__name__, 'message': str(exc),
                         'new_vectors': new, 'resume': 'Inspect failure; preserve all existing vectors'}, out/'status.json')
            raise


if __name__ == '__main__':
    main()
