"""Fixed V5 validation candidates and train-only language shortcut controls."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import torch
from .general import positive_mask
from .checkpoint import fingerprint


def question_type(question, source):
    """Heuristic labels use only question text/source, never the answer."""
    q = ' '.join(question.casefold().split())
    if source == 'vsr':
        return 'spatial_statement_verification'
    for name, pattern in [
        ('count', r'\bhow many\b|\bnumber of\b'),
        ('color', r'\bcolou?r\b'),
        ('shape', r'\bshape\b'),
        ('material', r'\bmaterial\b|\bmade of\b'),
        ('comparison', r'\bgreater\b|\bfewer\b|\bmore than\b|\bless than\b|\bhighest\b|\blowest\b|\bdifference\b'),
        ('location', r'\bwhere\b|\bposition\b|\blocated\b'),
        ('time', r'\bwhen\b|\bwhat year\b|\bwhich year\b|\bdate\b'),
        ('yes_no', r'^(is|are|does|do|can|was|were|has|have)\b'),
    ]:
        if re.search(pattern, q):
            return name
    return 'other'


def load_protocol(data):
    from scripts.train_general import pair, valid_records
    raw = (Path(data) / 'records.jsonl').read_bytes()
    all_rows = list(valid_records([json.loads(s) for s in raw.decode('utf-8').splitlines()]))
    sources = sorted({r['source'] for r in all_rows if r['split'] == 'train'})
    rows = sorted([r for r in all_rows if r['split'] == 'val' and r['source'] in sources],
                  key=lambda r: (r['source'], r['id']))
    items = []
    for row in rows:
        q, d, key = pair(row, random.Random(row['id']))
        text = random.Random(row['id']).choice(row['texts'])
        items.append({'id': row['id'], 'source': row['source'], 'task': row['task'],
                      'type': 'caption' if row['task'] == 'caption' else question_type(text['user'], row['source']),
                      'q': q, 'd': d, 'key': key, 'image_file_sha256': row['image_file_sha256']})
    manifest = {'protocol': 'v5-full-validation-controls-v1', 'data_sha256': hashlib.sha256(raw).hexdigest(),
                'items_sha256': fingerprint(items), 'queries': len(items),
                'selection': 'All validation records; pair(random.Random(record.id)); sorted source then id',
                'candidates': 'All same-source validation targets, including duplicate targets',
                'positives': 'Exact training positive_mask; normalized answers or overlapping image/caption keys',
                'ties': 'First candidate in record-id order, exactly as training validation',
                'types': 'Question/source-only heuristic v1; not official question-type annotation',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'training_pair_sha256': hashlib.sha256(Path('scripts/train_general.py').read_bytes()).hexdigest()}
    return manifest, items, all_rows


def summarize(items, correct):
    if len(items) != len(correct):
        raise ValueError('Prediction count mismatch')
    per_source = {}
    for source in sorted({r['source'] for r in items}):
        ix = [i for i, r in enumerate(items) if r['source'] == source]
        mask = positive_mask([items[i]['key'] for i in ix], 'cpu')
        types = {}
        for typ in sorted({items[i]['type'] for i in ix}):
            ids = [i for i in ix if items[i]['type'] == typ]
            types[typ] = {'queries': len(ids), 'R1': sum(bool(correct[i]) for i in ids) / len(ids)}
        per_source[source] = {'queries': len(ix), 'R1': sum(bool(correct[i]) for i in ix) / len(ix),
                              'candidate_count': len(ix),
                              'unique_positive_keys': len({fingerprint(items[i]['key']) for i in ix}),
                              'random_candidate_expected_R1': float(mask.float().mean()),
                              'per_question_type': types}
    return {'macro_r1': sum(r['R1'] for r in per_source.values()) / len(per_source),
            'micro_r1': sum(bool(x) for x in correct) / len(correct), 'per_source': per_source}


def score_vectors(items, q, d):
    if q.ndim != 2 or d.shape != q.shape or len(items) != len(q):
        raise ValueError('Invalid query/target shape')
    if not torch.isfinite(q).all() or not torch.isfinite(d).all():
        raise ValueError('Nonfinite vectors')
    correct = [False] * len(items)
    predictions = [None] * len(items)
    for source in sorted({r['source'] for r in items}):
        ix = [i for i, r in enumerate(items) if r['source'] == source]
        mask = positive_mask([items[i]['key'] for i in ix], 'cpu')
        ranks = (q[ix] @ d[ix].T).argmax(-1).tolist()
        for j, rank in enumerate(ranks):
            i = ix[j]
            correct[i] = bool(mask[j, rank])
            predictions[i] = {'id': items[i]['id'], 'candidate_id': items[ix[rank]]['id'], 'correct': correct[i]}
    return {**summarize(items, correct), 'predictions': predictions}


def majority_controls(items, all_rows):
    from scripts.train_general import pair
    counts, by_type = defaultdict(Counter), defaultdict(Counter)
    for row in all_rows:
        if row['split'] != 'train' or row['task'] == 'caption':
            continue
        _, _, key = pair(row, random.Random(row['id']))
        question = random.Random(row['id']).choice(row['texts'])['user']
        counts[row['source']][key] += 1
        by_type[(row['source'], question_type(question, row['source']))][key] += 1
    def most_common(counter):
        return min(counter, key=lambda k: (-counter[k], k))
    qa = [r for r in items if r['task'] != 'caption']
    results = {}
    for conditional in (False, True):
        hits, preds = [], []
        for row in qa:
            counter = by_type.get((row['source'], row['type'])) if conditional else None
            pred = most_common(counter or counts[row['source']])
            hits.append(pred == row['key'])
            # Do not publish original question/answer text in the report.
            preds.append({'id': row['id'], 'answer_key_sha256': hashlib.sha256(pred.encode()).hexdigest(),
                          'correct': hits[-1]})
        results['train_type_majority' if conditional else 'train_source_majority'] = {
            **summarize(qa, hits), 'predictions': preds,
            'rule': 'One deterministic pair per training image; lexicographic train-count ties; unseen types use source majority; absent answers count wrong'}
    return results


def paired_delta(items, full, control, samples=5000):
    import numpy as np
    a = {p['id']: p['correct'] for p in full['predictions']}
    b = {p['id']: p['correct'] for p in control['predictions']}
    rng = np.random.default_rng(20261011)
    results = {}
    for source in sorted({r['source'] for r in items}):
        ids = [r['id'] for r in items if r['source'] == source]
        delta = np.array([float(a[i]) - float(b[i]) for i in ids])
        boot = delta[rng.integers(0, len(ids), (samples, len(ids)))].mean(-1)
        results[source] = {'queries': len(ids), 'full_minus_text_only': float(delta.mean()),
                           'paired_bootstrap_95': np.quantile(boot, [.025, .975]).tolist()}
    return {'per_source': results, 'unit': 'query/image within each source; 5000 paired resamples; unadjusted intervals'}
