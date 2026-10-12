"""Recompute completed diagnostic predictions from byte-bound vector caches."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    os.chdir(ROOT)
    import torch
    from superembedding.checkpoint import atomic_json, fingerprint
    from superembedding.validation_protocol import load_protocol, score_vectors, paired_delta
    torch.set_num_threads(2)
    manifest, items, _ = load_protocol(STORE/'data_v51')
    results = {}
    for model in ['wemm2b', 'gme2b']:
        out = STORE/'paper_baselines_cpu_v3'/model
        prov = json.loads((out/'protocol.json').read_text())
        signature = fingerprint(prov)
        if manifest != prov['validation_protocol']:
            raise ValueError('Validation protocol changed')
        if any(sha(ROOT/p) != h for p, h in prov['implementation'].items()):
            raise ValueError('Implementation changed')
        checked = {}
        def vector(obj):
            hashes = []
            for p in obj.get('images', []):
                if p not in checked:
                    checked[p] = sha(p)
                hashes.append(checked[p])
            key = fingerprint({'object': obj, 'image_sha256': hashes, 'protocol': signature})
            saved = torch.load(out/'vectors'/(key+'.pt'), map_location='cpu', weights_only=True)
            v = saved['vector']
            if saved['fingerprint'] != key or not torch.isfinite(v).all() or abs(float(v.norm())-1) > 1e-4:
                raise ValueError('Corrupt vector')
            return v
        for label in ['vsr110', 'validation760']:
            path = out/(label+'.json')
            if not path.exists():
                continue
            original = json.loads(path.read_text())
            if original['provenance'] != prov or original['protocol_fingerprint'] != signature:
                raise ValueError('Result provenance differs')
            subset = [r for r in items if r['source'] == 'vsr'] if label == 'vsr110' else items
            q = torch.stack([vector(r['q']) for r in subset])
            d = torch.stack([vector(r['d']) for r in subset])
            qa = [r for r in subset if r['task'] == 'qa']
            ix = [i for i, r in enumerate(subset) if r['task'] == 'qa']
            t = torch.stack([vector({'text': r['q']['text']}) for r in qa])
            full = score_vectors(subset, q, d)
            qa_full = score_vectors(qa, q[ix], d[ix])
            text_only = score_vectors(qa, t, d[ix])
            recomputed = {'full': full, 'qa_full': qa_full, 'qa_text_only': text_only,
                          'paired_image_gain': paired_delta(qa, qa_full, text_only)}
            if any(original[k] != v for k, v in recomputed.items()):
                raise ValueError('Cached numerical replay differs')
            result = {'passed': True, 'result_sha256': sha(path), 'queries': len(subset),
                      'full_correct': sum(p['correct'] for p in full['predictions']),
                      'full_macro_r1': full['macro_r1'], 'qa_text_only_macro_r1': text_only['macro_r1'],
                      'scope': 'All cached predictions/scores replayed; not fresh inference of every case; seen diagnostic data.'}
            if label == 'vsr110':
                lookup = {r['id']: r for r in subset}
                for key, report in [('full', full), ('text_only', text_only)]:
                    recalls = {answer: sum(p['correct'] for p in report['predictions'] if lookup[p['id']]['key'] == answer)/
                               sum(r['key'] == answer for r in subset) for answer in sorted({r['key'] for r in subset})}
                    result[key+'_balanced_accuracy'] = sum(recalls.values())/len(recalls)
                    result[key+'_correct'] = sum(p['correct'] for p in report['predictions'])
            results[model+'/'+label] = result
        import pyarrow.parquet as pq
        science = STORE/'mmeb_scienceqa'
        dataset = science/'test.parquet'
        if sha(dataset) != prov['scienceqa_inputs']['parquet_sha256']:
            raise ValueError('ScienceQA table changed')
        for p, h in prov['scienceqa_inputs']['image_sha256'].items():
            if sha(science/p) != h:
                raise ValueError('ScienceQA image changed')
        rows = pq.read_table(dataset).to_pylist()
        for mode in ['full', 'text_only']:
            path = out/('scienceqa_'+mode+'.json')
            if not path.exists():
                continue
            original = json.loads(path.read_text())
            if original['provenance'] != prov or original['protocol_fingerprint'] != signature or original['dataset_sha256'] != sha(dataset):
                raise ValueError('ScienceQA result provenance differs')
            predictions = []
            for i, row in enumerate(rows):
                obj = {'text': row['qry_text'].replace('<|image_1|>', '')}
                if mode == 'full':
                    obj['images'] = [str(science/row['qry_img_path'])]
                q = vector(obj)
                d = torch.stack([vector({'text': s}) for s in row['tgt_text']])
                rank = int((q@d.T).argmax())
                predictions.append({'query': i, 'candidate_index': rank,
                                    'correct': row['tgt_text'][rank] == row['tgt_text'][0]})
            accuracy = sum(p['correct'] for p in predictions)/len(predictions)
            if predictions != original['predictions'] or accuracy != original['accuracy'] or len(predictions) != original['n']:
                raise ValueError('ScienceQA numerical replay differs')
            results[model+'/scienceqa_'+mode] = {'passed': True, 'queries': len(rows), 'accuracy': accuracy,
                'correct': sum(p['correct'] for p in predictions), 'result_sha256': sha(path),
                'scope': 'All cached predictions replayed; not fresh inference; pretraining contamination unknown.'}
    atomic_json({'results': results, 'completed_validation_tasks_checked': True, 'completed_scienceqa_tasks_checked': True,
                 'whole_project_complete': False}, ROOT/'artifacts/results/v5/paper_baseline_replay_audit.json')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
