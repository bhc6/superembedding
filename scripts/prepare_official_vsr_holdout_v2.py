"""Acquire pinned official VSR test; quarantine development-overlap candidates.

Never score models here. Candidates do not imply confirmed duplicate photos.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')
REPO = 'cambridgeltl/vsr_random'
REVISION = 'b2053328fafdd018ff56cf1dfa9643caaa4e69b8'
OUT = STORE/'official_vsr_holdout_v2'


def prepare():
    import httpx
    import numpy as np
    from PIL import Image
    from filelock import FileLock
    from scripts.audit_data_leakage import features, collect, normalize_text
    from superembedding.checkpoint import atomic_json
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT/'prepare.lock'), timeout=0):
        plan = {'protocol': 'official-vsr-random-holdout-audit-v2', 'dataset': REPO, 'revision': REVISION,
                'source_split': 'test', 'expected_rows': 2195,
                'checkpoint_rule': 'Fixed terminal Step2000; do not select checkpoints/prompts by this dataset.',
                'exclusion': 'Exact RGB or pHash<=6 AND dHash<=6 against all inventoried prior local image exposure; conservatively quarantine candidates before inference.',
                'text_overlap': 'Flag repeated spatial captions separately; they are not sufficient to classify image leakage.',
                'further_clearance': 'Crop/scene/entity review required. No automatic inference or release.',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'image_transport': 'Original official COCO HTTP image_link; HTTPS has hostname certificate mismatch. Persist byte hashes and image validation; no TLS verification bypass.',
                'metrics_used_for_filtering': False, 'release_eligible': False}
        path = OUT/'protocol.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Acquisition protocol changed')
        atomic_json(plan, path)
        client = httpx.Client(timeout=60, follow_redirects=True)

        def get(url):
            for attempt in range(5):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.content
                except httpx.HTTPError:
                    if attempt == 4:
                        raise RuntimeError('Source request failed after retries; URL omitted') from None
                    time.sleep(min(2**attempt, 10))

        source = OUT/'official_test.jsonl'
        if not source.exists():
            raw = get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/test.jsonl')
            temp = source.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(source)
        rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
        if len(rows) != 2195:
            raise ValueError('Official source row count differs')
        families, manifests = collect()
        history = []
        for group in families.values():
            history.extend(group)
        # Include records quarantined before v5.1, not merely the current manifest.
        original = STORE/'data/records.jsonl'
        history.extend(json.loads(s) for s in original.read_text(encoding='utf-8').splitlines())
        unique = {}
        for r in history:
            unique.setdefault(r['image'], []).append({'id': r['id'], 'source': r['source'], 'split': r['split']})
        inventory_path = OUT/'historical_inventory.json'
        inventory = {'manifests': manifests, 'original_manifest_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                     'images': unique, 'scope': 'All local experiment families and V5 external diagnostics; original V5 included.'}
        if inventory_path.exists() and json.loads(inventory_path.read_text()) != inventory:
            raise ValueError('Historical exposure inventory changed')
        atomic_json(inventory, inventory_path)
        # Reuse byte-keyed image features only after checking actual bytes.
        cached = STORE/'leakage_audit/features'
        hist = []
        atomic_json({'state': 'screening_historical_inventory', 'historical_images': len(unique), 'pid': os.getpid()}, OUT/'status.json')
        for image in unique:
            with Path(image).open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            old = cached/(digest+'.json')
            feat = json.loads(old.read_text()) if old.exists() else features(image)
            hist.append(feat)
        rgb = {}
        for image, feat in zip(unique, hist):
            rgb.setdefault(feat['rgb_sha256'], []).extend(unique[image])
        ph = np.array([f['phash'] for f in hist], dtype=np.uint64)
        dh = np.array([f['dhash'] for f in hist], dtype=np.uint64)
        history_paths = list(unique)
        # Spatial statement equality, independent of the many prompt templates.
        captions = set()
        import re
        for r in history:
            if r['source'] == 'vsr':
                for t in r.get('texts', []):
                    for c in re.findall(r'"([^"]+)"', t['user']):
                        captions.add(normalize_text(c))
        def acquire(item):
            i, row = item
            record = OUT/'rows'/f'{i:05d}.json'
            if record.exists():
                return json.loads(record.read_text())
            dest = OUT/'images'/row['image']
            if not dest.exists():
                raw = get(row['image_link'])
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Same photo can occur for multiple official statements.
                tmp = dest.with_suffix(f'.{i}.tmp'); tmp.write_bytes(raw); tmp.replace(dest)
            feat = features(str(dest))
            pd = np.bitwise_count(ph ^ np.uint64(feat['phash']))
            dd = np.bitwise_count(dh ^ np.uint64(feat['dhash']))
            hits = np.flatnonzero((pd <= 6) & (dd <= 6))
            candidates = [{'members': unique[history_paths[int(j)]],
                           'phash_distance': int(pd[j]), 'dhash_distance': int(dd[j])} for j in hits]
            exact = rgb.get(feat['rgb_sha256'], [])
            value = {'id': f'vsr_official_random:{i}', 'source_row': i, 'source_image': row['image'],
                     'image': str(dest), 'file_sha256': feat['file_sha256'], 'rgb_sha256': feat['rgb_sha256'],
                     'caption': row['caption'], 'label': row['label'], 'relation': row['relation'],
                     'caption_seen': normalize_text(row['caption']) in captions,
                     'exact_historical_matches': exact, 'near_historical_candidates': candidates,
                     'quarantined': bool(exact or candidates)}
            atomic_json(value, record)
            return value
        audit = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            for value in pool.map(acquire, enumerate(rows)):
                audit.append(value)
                if len(audit) % 25 == 0:
                    status = {'state': 'acquiring_and_screening', 'completed': len(audit), 'total': len(rows),
                              'quarantined_so_far': sum(r['quarantined'] for r in audit), 'pid': os.getpid(),
                              'elapsed_seconds': time.perf_counter()-started}
                    atomic_json(status, OUT/'status.json'); print(json.dumps(status), flush=True)
        result = {'protocol': plan, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'historical_inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                  'rows': len(audit), 'unique_images': len({r['rgb_sha256'] for r in audit}),
                  'exact_overlap_queries': sum(bool(r['exact_historical_matches']) for r in audit),
                  'near_candidate_queries': sum(bool(r['near_historical_candidates']) for r in audit),
                  'quarantined_queries': sum(r['quarantined'] for r in audit),
                  'retained_queries_pending_scene_review': sum(not r['quarantined'] for r in audit),
                  'caption_seen_queries': sum(r['caption_seen'] for r in audit),
                  'official_full_test_protocol_preserved': False,
                  'scope': 'Conservative development-disjoint subset candidate, not official full-test score. Review before any formal evaluation.',
                  'utc': datetime.now(timezone.utc).isoformat()}
        atomic_json(result, OUT/'audit.json')
        atomic_json(result, ROOT/'artifacts/results/v5/official_vsr_holdout_audit.json')
        atomic_json({'state': 'screened_review_required', **{k: result[k] for k in ['rows', 'quarantined_queries', 'retained_queries_pending_scene_review']}}, OUT/'status.json')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        prepare()
    except Exception as exc:
        from superembedding.checkpoint import atomic_json
        atomic_json({'state': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'resume': 'Rerun identical script after inspecting failure; source and completed rows retained'}, OUT/'status.json')
        raise
