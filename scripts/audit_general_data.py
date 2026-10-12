"""Seal a source manifest after checking image integrity and held-out task overlap."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

def hashes(path):
    with Image.open(path) as im:
        im=im.convert('RGB');sha=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        im.thumbnail((640,640),Image.Resampling.LANCZOS)
        small=np.asarray(im.convert('L').resize((9,8)))
        dhash=''.join('1' if v else '0' for v in (small[:,1:]>small[:,:-1]).flatten())
    return sha,dhash

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);args=ap.parse_args()
    root=Path(args.data);path=root/'records.jsonl'
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    test=Path('C:/Users/123/.cache/superembedding/v5/mmeb_scienceqa')
    if not (test/'metadata.json').exists():raise RuntimeError('Wait for full official test to finish downloading')
    with ThreadPoolExecutor(max_workers=8) as pool:known=list(pool.map(hashes,(test/'ScienceQA').glob('*.jpg')))
    if len(known)!=1000:raise ValueError('Expected 1000 official test images')
    exact={h[0] for h in known};near={h[1] for h in known}
    urban=Path('C:/Users/123/.cache/superembedding/v5/urban1k/images')
    with ThreadPoolExecutor(max_workers=8) as pool:urban_hashes=list(pool.map(hashes,urban.glob('*.jpg')))
    if len(urban_hashes)!=1000:raise ValueError('Expected 1000 Urban1k images')
    exact.update(h[0] for h in urban_hashes);near.update(h[1] for h in urban_hashes)
    previous=json.loads((root/'audit.json').read_text()) if (root/'audit.json').exists() else {}
    excluded=previous.get('test_overlap_excluded',[]).copy();kept=[]
    for row in rows:
        if row['original_rgb_sha256'] in exact or row['dhash'] in near:
            excluded.append({'id':row['id'],'reason':'official ScienceQA/Urban1k exact RGB or dHash overlap'});continue
        if hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()!=row['image_file_sha256']:
            raise RuntimeError('Local training image changed')
        if not row['texts'] or any(not isinstance(t.get('assistant'),str) or not isinstance(t.get('user'),str) for t in row['texts']):
            raise RuntimeError('Invalid text record')
        kept.append(row)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept)
    temp=path.with_suffix('.tmp');temp.write_text(payload,encoding='utf-8');temp.replace(path)
    report={'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(kept),
            'split_counts':dict(Counter(r['split'] for r in kept)),
            'source_split_counts':dict(Counter(r['source']+'/'+r['split'] for r in kept)),
            'supervision_pairs':sum(len(r['texts']) for r in kept),
            'test_overlap_excluded':excluded,'image_integrity':'all encoded files SHA256 verified',
            'near_duplicate_limit':'Exact dHash equality only; not exhaustive perceptual/semantic decontamination',
            'pretraining_contamination':'Unknown for all inherited backbones; no claim of clean pretraining.'}
    atomic_json(report,root/'audit.json');atomic_json(report,'artifacts/results/v5/data_audit.json')
    metadata=json.loads((root/'metadata.json').read_text())
    metadata.update(manifest_sha256=report['manifest_sha256'],records=len(kept),test_overlap_audited=True)
    atomic_json(metadata,root/'metadata.json')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
