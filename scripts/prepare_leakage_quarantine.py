"""Versioned conservative CUB/DTD candidate quarantine, never a clean-data certificate."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superembedding.leakage_guard import digest, ROOT
from superembedding.checkpoint import atomic_json


def quarantine(rows, pairs):
    lookup = {r['id']: r for r in rows}
    parent = {key: key for key in lookup}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for pair in pairs:
        a, b = find(pair['a']), find(pair['b'])
        parent[b] = a
    groups = defaultdict(list)
    for key in lookup:
        groups[find(key)].append(key)
    priority = {'test': 0, 'val': 1, 'train': 2}
    removed = set()
    components = []
    for keys in groups.values():
        splits = {lookup[key]['split'] for key in keys}
        if len(splits) < 2:
            continue
        keep = min(splits, key=priority.get)
        excluded = sorted(k for k in keys if lookup[k]['split'] != keep)
        removed.update(excluded)
        components.append({'members': sorted(keys), 'retained_split': keep, 'quarantined_ids': excluded})
    return [dict(r) for r in rows if r['id'] not in removed], components


def immutable(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Refusing to overwrite versioned artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)


def main():
    audit_path = ROOT/'artifacts/results/v5/data_leakage_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for name in ['cub_attributes', 'dtd']:
        source = ROOT/'data'/name
        if digest(source/'records.jsonl') != audit['provenance']['manifests'][name]['sha256']:
            raise ValueError('Stale source manifest')
        rows = [json.loads(s) for s in (source/'records.jsonl').read_text(encoding='utf-8').splitlines()]
        candidates = audit['families'][name]['perceptual_candidates'] + audit['families'][name]['exact_cross_split_pairs']
        kept, components = quarantine(rows, candidates)
        ids = {r['id'] for r in kept}
        retained = {r['id']: r for r in kept}
        assert not any(p['a'] in ids and p['b'] in ids and retained[p['a']]['split'] != retained[p['b']]['split'] for p in candidates)
        assert {r['id'] for r in rows if r['split']=='test'} == {r['id'] for r in kept if r['split']=='test'}
        for r in kept:
            r['image'] = str((source/r['image']).resolve())
        train = sorted((r for r in kept if r['split']=='train'), key=lambda r:r['train_rank'])
        for rank,r in enumerate(train):
            r['train_rank'] = rank
        raw = ''.join(json.dumps(r,sort_keys=True)+'\n' for r in kept).encode()
        out = ROOT/'data'/(name+'_leakage_v2_draft')
        immutable(out/'records.jsonl', raw)
        meta = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
        meta.update(count=len(kept), splits=dict(Counter(r['split'] for r in kept)),
                    scales={str(p):max(1,round(len(train)*p/100)) for p in (25,50,100)},
                    manifest_sha256=digest(out/'records.jsonl'),
                    labels_observed={s:{k:sum(bool(r['labels'][k]) for r in kept if r['split']==s)
                                       for k in meta['label_keys']} for s in ['train','val','test']},
                    protocol='Draft conservative connected-component quarantine; priority test > val > train; not original official protocol.',
                    leakage_clearance=False, historical_checkpoint_resume_allowed=False)
        immutable(out/'metadata.json', (json.dumps(meta,indent=2)+'\n').encode())
        result = {'dataset':name, 'output':str(out), 'status':'draft_not_cleared_for_training_or_claims',
                  'source_sha256':digest(source/'records.jsonl'), 'manifest_sha256':digest(out/'records.jsonl'),
                  'audit_sha256':digest(audit_path), 'policy':'All screened pairs treated conservatively as grouping edges; not all are confirmed duplicates.',
                  'components':components, 'quarantined_by_split':dict(Counter(r['split'] for r in rows if r['id'] not in ids)),
                  'retained_by_split':meta['splits'], 'test_ids_unchanged':True,
                  'screened_cross_split_edges_remaining':0,
                  'further_work':['Crop/scene review beyond hash threshold',
                                  'Fresh trainable head initialization; never resume contaminated heads',
                                  'Rerun all matched baselines; unchanged frozen encoders may reuse verified per-image features',
                                  'Seen test remains diagnostic; new unseen confirmation needed'],
                  'predictions_used_for_filtering':False}
        immutable(ROOT/'artifacts/data'/out.name/'quarantine.json', (json.dumps(result,indent=2)+'\n').encode())
        print(json.dumps({k:result[k] for k in ['dataset','quarantined_by_split','retained_by_split','status']}))


if __name__ == '__main__':
    main()
