"""Same candidates as V5 training, with restartable baseline and QA controls."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['majority', 'raw_cached', 'raw', 'general', 'qwen_api', 'qwen2b'], required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--checkpoint')
    ap.add_argument('--data', default='D:/DEMO/superembedding-storage/superembedding/v5/data_v51')
    ap.add_argument('--features', default='D:/DEMO/superembedding-storage/superembedding/v5/general_ablation_features')
    ap.add_argument('--model', default='D:/DEMO/superembedding-storage/official_models/Qwen--Qwen3-VL-Embedding-2B')
    ap.add_argument('--cache', default='D:/DEMO/superembedding-storage/superembedding/v5/validation_controls')
    ap.add_argument('--max-new', type=int, default=0, help='Smoke limit; never emit complete metrics until all requested vectors exist')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('MSO_FLA', '0')
    os.environ.setdefault('MSO_BRANCH_CHUNK', '4')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from superembedding.checkpoint import atomic_json, atomic_save, fingerprint
    from superembedding.validation_protocol import load_protocol, majority_controls, score_vectors, paired_delta
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol, items, all_rows = load_protocol(args.data)
    qa_indices = [i for i, r in enumerate(items) if r['task'] != 'caption']
    qa = [items[i] for i in qa_indices]
    output = Path('artifacts/results/v5/validation_controls')
    output.mkdir(parents=True, exist_ok=True)
    if args.backend == 'majority':
        result = {'protocol': protocol, 'controls': majority_controls(items, all_rows)}
        atomic_json(result, output/'majority.json')
        print(json.dumps({k: v['macro_r1'] for k, v in result['controls'].items()}), flush=True)
        return
    prov = {'protocol': protocol, 'backend': args.backend, 'device': args.device,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.checkpoint:
        prov['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if args.backend == 'general' and not args.checkpoint:
        raise ValueError('Complete inference bundle required')
    if args.backend == 'raw_cached':
        root = Path(args.features)
        feature_prov = json.loads((root/'provenance.json').read_text(encoding='utf-8'))
        sig = fingerprint(feature_prov)
        status = json.loads((root/'status.json').read_text(encoding='utf-8'))
        if status.get('state') != 'completed' or status.get('fingerprint') != sig:
            raise ValueError('Features incomplete')
        if feature_prov['data_sha256'] != protocol['data_sha256']:
            raise ValueError('Data mismatch')
        positions = {key: i for i, key in enumerate(feature_prov['ids'])}
        vectors = []
        for r in items:
            saved = torch.load(root/'shards'/f"{positions[r['id']]:05d}.pt", map_location='cpu', weights_only=True)
            if saved['id'] != r['id'] or saved['fingerprint'] != sig or saved['key'] != r['key']:
                raise ValueError('Feature identity/key mismatch')
            vectors.append([F.normalize(F.normalize(saved[k].float(), dim=-1).flatten(), dim=-1) for k in ('q','d')])
        q = torch.stack([v[0] for v in vectors]); d = torch.stack([v[1] for v in vectors])
        result = {'provenance': {**prov, 'feature_provenance': feature_prov, 'dtype': 'float32'},
                  'full': score_vectors(items, q, d),
                  'limitation': 'Same queries/candidates/metric; CPU float32, while live training validation uses CUDA bfloat16. Not a precision-controlled gain estimate.'}
        atomic_json(result, output/'raw_cached.json')
        print(json.dumps({'backend': args.backend, 'macro_r1': result['full']['macro_r1']}), flush=True)
        return
    if args.backend == 'qwen_api':
        from superembedding.baselines import MODELS, ENDPOINT
        prov.update(model=MODELS['qwen3_vl'], endpoint=ENDPOINT,
                    precision='Provider-managed', preprocessing='Original image bytes, provider preprocessing',
                    revision='Mutable API alias; cached vectors and timestamps define this run, NOT identified as local 2B weights')
    elif args.backend == 'qwen2b':
        official = Path(args.model)
        code = official/'scripts/qwen3_vl_embedding.py'
        prov.update(model_download=json.loads((official/'download_verified.json').read_text(encoding='utf-8')),
                    official_code_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
                    dtype='float32' if args.device == 'cpu' else 'bfloat16',
                    max_pixels=65536, max_length=8192, instruction="Represent the user's input.",
                    scope='Pinned official model and pooling, 65536-pixel resource-constrained track; not official default resolution',
                    vision_errors='Fail closed; never silently replace image inputs with NULL')
    else:
        prov.update(dtype='float32' if args.device == 'cpu' else 'bfloat16', max_pixels=65536,
                    encoding='GeneralOmniEncoder; fixed four directions',
                    adapter_sha256=hashlib.sha256(Path('models/OmniJev-0.8B-v1.1/adapter_model.safetensors').read_bytes()).hexdigest())
    tag = args.backend + ('_'+prov['checkpoint_sha256'][:12] if args.checkpoint else '') + '_' + args.device
    out = Path(args.cache)/tag
    out.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/'evaluation.lock'), timeout=0):
        path = out/'provenance.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != prov:
            raise ValueError('Evaluation provenance changed; use a new cache directory')
        atomic_json(prov, path)
        (out/'implementation.py').write_bytes(Path(__file__).read_bytes())
        # Verify each source image before sending/encoding any data.
        for r in items:
            paths = r['q'].get('images', []) + r['d'].get('images', [])
            if len(paths) != 1 or hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest() != r['image_file_sha256']:
                raise ValueError('Validation image changed')
        def key(obj):
            return fingerprint({'object': obj, 'provenance': prov})
        objects = {}
        for r in items:
            for obj in (r['q'], r['d']):
                objects[key(obj)] = obj
        for r in qa:
            obj = {'text': r['q']['text']}
            objects[key(obj)] = obj
        def load_vector(k):
            saved = torch.load(out/'vectors'/f'{k}.pt', map_location='cpu', weights_only=True)
            v = saved['vector']
            if saved['fingerprint'] != k or v.ndim != 1 or not torch.isfinite(v).all() or abs(float(v.norm())-1) > 1e-4:
                raise ValueError('Invalid cached vector')
            return v
        pending = []
        for k, obj in objects.items():
            if (out/'vectors'/f'{k}.pt').exists():
                load_vector(k)
            else:
                pending.append((k,obj))
        if args.max_new:
            pending = pending[:args.max_new]
        client = None
        model = None
        if pending and args.backend == 'qwen_api':
            import httpx
            from scripts.extract_baselines import api_key
            client = httpx.Client(headers={'Authorization': 'Bearer '+api_key()}, timeout=120)
            def encode(batch):
                contents = []
                for obj in batch:
                    content = {'text': obj['text']} if obj.get('text') else {}
                    if obj.get('images'):
                        path = Path(obj['images'][0])
                        with Image.open(path) as im:
                            mime = Image.MIME.get(im.format, 'image/jpeg')
                        content['image'] = f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()
                    contents.append(content)
                body = {'model': prov['model']['model'], 'input': {'contents': contents}, 'parameters': prov['model']['parameters']}
                for attempt in range(7):
                    try:
                        response = client.post(ENDPOINT, json=body)
                    except httpx.HTTPError:
                        if attempt == 6:
                            raise RuntimeError('API transport retries exhausted') from None
                        time.sleep(min(2**attempt,30)); continue
                    if response.status_code in {429,500,502,503,504}:
                        time.sleep(min(2**attempt,30)); continue
                    if response.status_code != 200:
                        raise RuntimeError(f'API HTTP {response.status_code}')
                    data = response.json(); embeddings = data.get('output',{}).get('embeddings',[])
                    if sorted(e['index'] for e in embeddings) != list(range(len(batch))):
                        raise ValueError('API embedding indices mismatch')
                    return torch.tensor([e['embedding'] for e in sorted(embeddings,key=lambda e:e['index'])]), {
                        'request_id': data.get('request_id'), 'usage': data.get('usage'),
                        'utc': datetime.now(timezone.utc).isoformat()}
                raise RuntimeError('API server retries exhausted')
        elif pending and args.backend == 'qwen2b':
            spec = importlib.util.spec_from_file_location('pinned_qwen_embedding', code)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            model = module.Qwen3VLEmbedder(str(official), max_pixels=65536, max_length=8192,
                dtype=torch.float32 if args.device=='cpu' else torch.bfloat16, local_files_only=True)
            # The official helper catches vision errors and encodes NULL. Use its
            # formatting/vision/processor/pooling with explicit fail-closed handling.
            def encode(batch):
                conversations = []
                for obj in batch:
                    image = None
                    if obj.get('images'):
                        with Image.open(obj['images'][0]) as im:
                            image = im.convert('RGB')
                    conversations.append(model.format_model_input(text=obj.get('text'),image=image))
                images, videos, vision_kwargs = module.process_vision_info(conversations, image_patch_size=16,
                    return_video_metadata=True, return_video_kwargs=True)
                if videos:
                    raise ValueError('Unexpected video')
                if sum(bool(o.get('images')) for o in batch) != len(images or []):
                    raise ValueError('Vision input lost')
                rendered = model.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
                inputs = model.processor(text=rendered, images=images, videos=None, video_metadata=None,
                    truncation=True,max_length=8192,padding=True,do_resize=False,return_tensors='pt',**vision_kwargs)
                inputs = {k:v.to(model.model.device) for k,v in inputs.items()}
                with torch.no_grad():
                    outputs = model.forward(inputs)
                    vectors = model._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])
                return vectors.float().cpu(), {'pooling': 'Official last nonpadding token', 'vision_validated': True}
        elif pending:
            from superembedding.general import GeneralOmniEncoder, GENERAL_BASIS
            from superembedding.omnijev import load_omnijev
            if args.backend == 'general':
                from superembedding.general_export import load_bundle_encoder
                model, _ = load_bundle_encoder(args.checkpoint)
            else:
                model = GeneralOmniEncoder(load_omnijev('models/OmniJev-0.8B-v1.1', 'models/Qwen3.5-0.8B',max_pixels=65536),
                                           GENERAL_BASIS,mode='raw',tune_lora=False).eval()
            def encode(batch):
                vectors = []
                with torch.no_grad():
                    for obj in batch:
                        if args.backend == 'raw':
                            u, _ = model.extract_features(obj)
                            vectors.append(F.normalize(F.normalize(u.float(),dim=-1).flatten(),dim=-1).cpu())
                        else:
                            vectors.append(model(obj).embedding.cpu())
                return torch.stack(vectors), {'max_pixels': 65536}
        batch_size = 8 if args.backend == 'qwen_api' else 1
        jobs = [pending[i:i+batch_size] for i in range(0,len(pending),batch_size)]
        def run(batch):
            vectors, info = encode([obj for _,obj in batch])
            if vectors.ndim != 2 or len(vectors) != len(batch) or not torch.isfinite(vectors).all() or (vectors.norm(dim=-1)<1e-8).any():
                raise ValueError('Invalid model vectors')
            for (k,_),v in zip(batch,F.normalize(vectors.float(),dim=-1)):
                atomic_save({'vector':v.cpu(),'fingerprint':k,'info':info}, out/'vectors'/f'{k}.pt')
        started = time.perf_counter()
        completed = len(objects) - (len(pending) if not args.max_new else sum(not (out/'vectors'/f'{k}.pt').exists() for k in objects))
        def progress(i):
            state = {'state':'encoding','backend':tag,'pid':os.getpid(),'batches_completed':i,'batches_this_launch':len(jobs),
                     'total_unique_objects':len(objects),'cached_at_start':completed,'seconds':time.perf_counter()-started}
            atomic_json(state,out/'status.json')
            if i == 1 or i%25 == 0:
                print(json.dumps(state),flush=True)
        try:
            if args.backend == 'qwen_api':
                with ThreadPoolExecutor(args.workers) as pool:
                    for i,_ in enumerate(pool.map(run,jobs),1): progress(i)
            else:
                for i,job in enumerate(jobs,1): run(job); progress(i)
        finally:
            if client is not None: client.close()
        if any(not (out/'vectors'/f'{k}.pt').exists() for k in objects):
            atomic_json({'state':'smoke_complete','total_unique_objects':len(objects)},out/'status.json')
            return
        q = torch.stack([load_vector(key(r['q'])) for r in items])
        d = torch.stack([load_vector(key(r['d'])) for r in items])
        qt = torch.stack([load_vector(key({'text':r['q']['text']})) for r in qa])
        full = score_vectors(items,q,d)
        qa_full = score_vectors(qa,q[qa_indices],d[qa_indices])
        text_only = score_vectors(qa,qt,d[qa_indices])
        result = {'provenance':prov,'full':full,'qa_full':qa_full,'qa_text_only':text_only,
                  'qa_paired_delta':paired_delta(qa,qa_full,text_only),
                  'control':'Only query images removed; identical frozen target vectors and candidates. Caption sources excluded from text-only QA score.',
                  'test_based_selection':False}
        atomic_json(result,output/f'{tag}.json')
        atomic_json({'state':'completed','result':str(output/f'{tag}.json'),'unique_objects':len(objects)},out/'status.json')
        print(json.dumps({'backend':tag,'macro_r1':full['macro_r1'],'qa_full':qa_full['macro_r1'],'qa_text_only':text_only['macro_r1']}),flush=True)


if __name__ == '__main__':
    main()
