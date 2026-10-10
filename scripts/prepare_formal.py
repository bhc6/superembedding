"""New held-out subset: exclude the development set and dHash near duplicates."""
import hashlib
import io
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def dhash(image):
    pixels = list(image.convert('L').resize((9, 8)).get_flattened_data())
    return sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


def main():
    dev = ROOT/'data/fashion600'
    out = ROOT/'data/fashion2400'
    if (out/'records.jsonl').exists():
        print('Existing frozen manifest retained.'); return
    meta = json.loads((dev/'metadata.json').read_text())
    old = [json.loads(s) for s in (dev/'records.jsonl').read_text().splitlines()]
    hashes = {r['image_sha256'] for r in old}
    ids = {r['id'] for r in old}
    perceptual = [dhash(Image.open(dev/r['image'])) for r in old]
    rows = []
    shards = ['data/train-00000-of-00002-6cff4c59f91661c3.parquet',
              'data/train-00001-of-00002-bb459e5ac5f01e71.parquet']
    for filename in shards:
        local = hf_hub_download(meta['dataset'], filename, repo_type='dataset',
                                revision=meta['revision'], local_dir=str(dev/'source'))
        rows.extend(pq.read_table(local, columns=['id','baseColour','articleType','gender','image']).to_pylist())
    rng = random.Random(20261011)
    rng.shuffle(rows)
    groups = defaultdict(list)
    stats = Counter()
    for r in rows:
        if r['baseColour'] not in {'Black','White','Blue','Red','Green','Grey','Brown','Pink'} or r['articleType'] not in {'Tshirts','Shirts','Casual Shoes','Sports Shoes','Tops','Handbags'} or r['gender'] not in {'Men','Women'}:
            continue
        if str(r['id']) in ids:
            stats['excluded_id'] += 1; continue
        image = Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')
        digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        ph = dhash(image)
        if digest in hashes or any((ph ^ p).bit_count() <= 4 for p in perceptual):
            stats['excluded_exact_or_dhash'] += 1; continue
        hashes.add(digest); ids.add(str(r['id'])); perceptual.append(ph)
        r['image_sha256'] = digest
        r['dhash'] = f'{ph:016x}'
        groups[(r['baseColour'],r['articleType'],r['gender'])].append(r)
    groups = {k:v for k,v in groups.items() if len(v)>=20}
    selected = defaultdict(list)
    keys = sorted(groups)
    while sum(map(len, selected.values())) < 2400:
        progress = False
        for k in keys:
            if groups[k] and sum(map(len, selected.values())) < 2400:
                selected[k].append(groups[k].pop()); progress=True
        if not progress:
            raise RuntimeError('Insufficient unique images for frozen 2400-image protocol')
    (out/'images').mkdir(parents=True, exist_ok=True)
    records=[]
    for k, group in sorted(selected.items()):
        n=len(group); nv=max(2,n//5); nt=nv
        for j,r in enumerate(group):
            split='train' if j<n-nv-nt else ('val' if j<n-nt else 'test')
            relative=f"images/{r['id']}.jpg"
            (out/relative).write_bytes(r['image']['bytes'])
            records.append({'id':str(r['id']),'image':relative,'split':split,
                            'labels':dict(zip(meta['label_keys'], k)),
                            'image_sha256':r['image_sha256'],'dhash':r['dhash']})
    content=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)
    (out/'records.jsonl').write_text(content,encoding='utf-8')
    meta.update(count=len(records),shards=shards,seed=20261011,
                splits=dict(Counter(r['split'] for r in records)),
                protocol='Joint-stratified held-out product split, development IDs and exact/dHash-distance<=4 duplicates excluded before splitting; seen values/combinations; dHash is heuristic, not a semantic duplicate guarantee.',
                manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
                filtering=dict(stats),joint_strata=len(selected))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))


if __name__=='__main__': main()
