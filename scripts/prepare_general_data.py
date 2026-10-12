"""Bounded, resumable train-only source acquisition; no signed URLs in manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import httpx
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

REPO = 'HuggingFaceM4/the_cauldron'
REV = '847a98a779b1652d65111daf20c972dfcd333605'
SOURCES = ['localized_narratives', 'textcaps', 'screen2words', 'chart2text',
           'docvqa', 'chartqa', 'clevr', 'vsr']
CAPTION = set(SOURCES[:4])


def get(client, url, **kwargs):
    for attempt in range(12):
        try:
            response = client.get(url, **kwargs)
            if response.status_code==429:
                if attempt==11:raise RuntimeError('Dataset service rate limit persists')
                time.sleep(min(45,max(10,int(response.headers.get('retry-after','30')))))
                continue
            response.raise_for_status()
            return response
        except (httpx.HTTPError, OSError):
            if attempt == 11:
                raise RuntimeError('Download failed after retries (URL redacted)') from None
            time.sleep(min(2 ** attempt, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--per-source', type=int, default=1024)
    ap.add_argument('--holdout-per-source', type=int, default=96)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--download-only',action='store_true')
    args = ap.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    (root/'images').mkdir(exist_ok=True); (root/'rows').mkdir(exist_ok=True)
    all_sources = SOURCES + ['ai2d', 'scienceqa']
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for source in args.sources or all_sources:
            if source not in all_sources:raise ValueError('Unknown source')
            params = dict(dataset=REPO, config=source, split='train')
            first = get(client, 'https://datasets-server.huggingface.co/rows', params={**params,'offset':0,'length':1}).json()
            total = first['num_rows_total']
            count = args.per_source if source in SOURCES else args.holdout_per_source
            rng = random.Random('v5-source-selection-20261011-' + source)
            # Randomly distributed blocks; stable nested ordering at growing scales.
            blocks = list(range((total + 31)//32)); rng.shuffle(blocks)
            selected = []
            for block in blocks:
                selected += list(range(block*32, min(total, (block+1)*32)))
                if len(selected) >= count: break
            selected = selected[:count]
            def save_row(item):
                idx = item['row_idx']; row = item['row']; destination = root/'rows'/f'{source}_{idx}.json'
                if destination.exists(): return
                if len(row['images']) != 1 or not row['texts']:
                    atomic_json({'rejected': 'requires one image and nonempty text'}, destination); return
                url = row['images'][0]['src']
                if f'/--/{REV}/--/' not in url:
                    raise RuntimeError('Source revision changed')
                raw = get(client, url).content
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert('RGB')
                    original_hash = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                    im.thumbnail((640,640), Image.Resampling.LANCZOS)
                    visual_hash = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                    image_path = root/'images'/f'{visual_hash}.jpg'
                    if not image_path.exists():
                        temp = image_path.with_suffix(f'.{source}_{idx}.tmp')
                        im.save(temp, format='JPEG', quality=90)
                        temp.replace(image_path)
                    # Cheap near-duplicate key; conservative cross-split exclusion below.
                    import numpy as np
                    small = np.asarray(im.convert('L').resize((9,8)))
                    dhash = ''.join('1' if v else '0' for v in (small[:,1:] > small[:,:-1]).flatten())
                atomic_json({'id':f'{source}:{idx}', 'source':source, 'source_row':idx,
                    'source_revision':REV, 'original_rgb_sha256':original_hash,
                    'image':str(image_path.resolve()), 'image_file_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    'dhash':dhash, 'texts':row['texts']}, destination)
            chunks=[]
            for idx in selected:
                if not chunks or len(chunks[-1])==32 or idx!=chunks[-1][-1]+1:
                    chunks.append([])
                chunks[-1].append(idx)
            done=0
            for subset in chunks:
                done+=len(subset)
                if all((root/'rows'/f'{source}_{i}.json').exists() for i in subset): continue
                data = get(client,'https://datasets-server.huggingface.co/rows',
                           params={**params,'offset':subset[0],'length':len(subset)}).json()
                if [r['row_idx'] for r in data['rows']]!=subset:
                    raise RuntimeError('Source row selection mismatch')
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(save_row, data['rows']))
                print(json.dumps({'source':source,'downloaded':done,'planned':count}),flush=True)
                time.sleep(2)
            atomic_json(selected, root/f'{source}_selection.json')
    if args.download_only:return
    records = []
    for source in all_sources:
        for order, idx in enumerate(json.loads((root/f'{source}_selection.json').read_text())):
            row = json.loads((root/'rows'/f'{source}_{idx}.json').read_text(encoding='utf-8'))
            if 'rejected' in row: continue
            split = 'ood' if source not in SOURCES else ('val' if int(hashlib.sha256(row['original_rgb_sha256'].encode()).hexdigest()[:8],16)%10 == 0 else 'train')
            records.append({**row,'split':split,'order':order,'task':'caption' if source in CAPTION else 'qa'})
    # All sources split by original image, then remove train/val duplicates of OOD;
    # identical 64-bit dHash is a conservative proxy, not full perceptual dedup.
    priority = {'train':0,'val':1,'ood':2}; groups = {}
    for row in records:
        for key in (row['original_rgb_sha256'], row['dhash']):
            groups[key] = max(groups.get(key,0), priority[row['split']])
    clean = [r for r in records if all(groups[k] == priority[r['split']] for k in (r['original_rgb_sha256'],r['dhash']))]
    path = root/'records.jsonl'
    payload = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean)
    path.with_suffix('.tmp').write_text(payload,encoding='utf-8'); path.with_suffix('.tmp').replace(path)
    atomic_json({'repo':REPO,'revision':REV,'train_sources':SOURCES,'ood_sources':all_sources[-2:],
        'original_split':'train only; OOD here means held-out source, not official test',
        'per_source':args.per_source,'records':len(clean),'excluded_cross_split_duplicates':len(records)-len(clean),
        'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'redistribution':'Recipes and source identifiers only. Upstream image/text licenses apply.',
        'dedup_limit':'Exact RGB and exact 64-bit dHash only; no claim of exhaustive near-duplicate exclusion.'},root/'metadata.json')
    print(json.dumps({'complete':True,'records':len(clean)}),flush=True)


if __name__ == '__main__': main()
