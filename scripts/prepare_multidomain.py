"""Version 3 research manifests. No model predictions are used for labels/splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import pyarrow.parquet as pq
import numpy as np
from scripts.prepare_formal import dhash

ROOT=Path(__file__).resolve().parents[1]
SEED=20261012

def image_hash(path):
    with Image.open(path) as im:
        im=im.convert('RGB')
        return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def save(name, rows, meta):
    out=ROOT/'data'/name; out.mkdir(parents=True,exist_ok=True)
    # Exact duplicate policy: official test > validation > train. Preserve one
    # image per decoded hash, also within splits; record all removals.
    seen=set(); kept=[]; removed=[]
    for r in sorted(rows,key=lambda r:({'test':0,'val':1,'train':2}[r['split']],r['id'])):
        if r['image_sha256'] in seen: removed.append(r['id']); continue
        seen.add(r['image_sha256']); kept.append(r)
    train=[r for r in kept if r['split']=='train']
    # Interleave stratum quantiles to preserve prevalence as data volume grows.
    groups=defaultdict(list); rng=random.Random(SEED)
    for r in train: groups[tuple(r['labels'][meta['label_keys'][0]])].append(r)
    for group in groups.values(): rng.shuffle(group)
    ordered=[r for _,_,r in sorted(((j+.5)/len(group),r['id'],r)
             for group in groups.values() for j,r in enumerate(group))]
    for i,r in enumerate(ordered): r['train_rank']=i
    counts={s:sum(r['split']==s for r in kept) for s in ('train','val','test')}
    meta.update(seed=SEED,count=len(kept),splits=counts,
                scales={str(p):max(1,round(len(train)*p/100)) for p in (25,50,100)},
                exact_duplicate_removed_ids=removed,label_format='lists; empty list means unobserved',
                labels_observed={s:{k:sum(bool(r['labels'][k]) for r in kept if r['split']==s)
                    for k in meta['label_keys']} for s in counts})
    kept.sort(key=lambda r:r['id'])
    content=''.join(json.dumps(r,sort_keys=True)+'\n' for r in kept)
    meta['manifest_sha256']=hashlib.sha256(content.encode()).hexdigest()
    if (out/'records.jsonl').exists() and (out/'records.jsonl').read_text(encoding='utf-8')!=content:
        raise ValueError(f'Refusing to change frozen {name} manifest; use a new dataset/version directory')
    (out/'records.jsonl').write_text(content,encoding='utf-8',newline='\n')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    artifact=ROOT/'artifacts'/'data'/name; artifact.mkdir(parents=True,exist_ok=True)
    for fn in ('metadata.json','records.jsonl'): (artifact/fn).write_bytes((out/fn).read_bytes())
    print(json.dumps({'dataset':name,**meta},indent=2),flush=True)

def cub():
    root=ROOT/'data/official/CUB_200_2011'
    names={int(s.split()[0]):s.split()[1] for s in (root/'images.txt').read_text().splitlines()}
    split={int(a):int(b) for a,b in (s.split() for s in (root/'train_test_split.txt').read_text().splitlines())}
    # attributes.txt is distributed in the outer archive directory.
    paths=[root/'attributes/attributes.txt',root/'attributes.txt',root.parent/'attributes.txt']
    attfile=next((p for p in paths if p.exists()),None)
    if attfile is None: raise FileNotFoundError('Official CUB attributes.txt required')
    attr={int(s.split()[0]):s.split()[1] for s in attfile.read_text().splitlines()}
    groups={'primary_color':'has_primary_color','bill_shape':'has_bill_shape','wing_shape':'has_wing_shape'}
    lookup={a:(k,v.split('::')[1]) for a,v in attr.items() for k,prefix in groups.items() if v.startswith(prefix+'::')}
    labels={i:{k:[] for k in groups} for i in names}
    for line in (root/'attributes/image_attribute_labels.txt').read_text().splitlines():
        fields=line.split(); i,a,present,certainty=map(int,fields[:4])
        if a in lookup and present==1 and certainty>=3:
            key,value=lookup[a]; labels[i][key].append(value)
    rng=random.Random(SEED); val=set(); classes=defaultdict(list)
    for i in names:
        if split[i]: classes[names[i].split('/')[0]].append(i)
    for ids in classes.values():
        rng.shuffle(ids); val.update(ids[:max(2,round(len(ids)*.2))])
    rows=[]
    for i,path in names.items():
        image=root/'images'/path
        rows.append({'id':f'cub_{i:05d}','image':'../official/CUB_200_2011/images/'+path,
                     'image_sha256':image_hash(image),'labels':labels[i],
                     'species':path.split('/')[0],
                     'split':'test' if not split[i] else ('val' if i in val else 'train')})
    save('cub_attributes',rows,{'dataset':'CUB-200-2011','revision':'65de6-vp158',
        'label_keys':list(groups),'directions':['the primary color of the bird','the shape of the bird bill','the shape of the bird wings'],
        'instruction':'Extract visible evidence about the pictured bird for the specified attribute direction.',
        'label_policy':'Retain all positive attributes with human certainty >=3 (probably/definitely); absent high-confidence positives mean unobserved, not a negative class.',
        'protocol':'Official train/test; species-stratified 20% official train for validation. Within-split retrieval; pair relevant iff at least one shared observed value in every requested attribute.',
        'license':'Images restricted to non-commercial research and education; no image redistribution.',
        'source':'https://www.vision.caltech.edu/datasets/cub_200_2011/'})

def dtd():
    root=ROOT/'data/official/dtd'; rows=[]
    joint={p[0]:p[1:] for p in (s.split() for s in (root/'labels/labels_joint_anno.txt').read_text().splitlines())}
    for split in ('train','val','test'):
        for path in (root/f'labels/{split}1.txt').read_text().splitlines():
            rows.append({'id':'dtd_'+path.replace('/','_').removesuffix('.jpg'),
                         'image':'../official/dtd/images/'+path,'split':split,
                         'image_sha256':image_hash(root/'images'/path),
                         'labels':{'texture':[path.split('/')[0]]},'joint_texture_labels':joint[path]})
    save('dtd',rows,{'dataset':'Describable Textures Dataset','revision':'r1.0.1',
        'label_keys':['texture'],'directions':['the visible texture and repeated surface pattern'],
        'instruction':'Extract evidence about the pictured surface for the specified texture direction.',
        'protocol':'Official split 1 train/val/test. Primary relevance is equal key texture; secondary metric uses shared human joint texture annotations. This is not the official 10-split classification benchmark.',
        'license':'Official website provides data for research purposes; no image redistribution.',
        'source':'https://www.robots.ox.ac.uk/~vgg/data/dtd/'})

def fashion():
    old=[]; meta=json.loads((ROOT/'data/fashion2400/metadata.json').read_text())
    for name in ('fashion600','fashion2400'):
        old.extend(json.loads(s) for s in (ROOT/f'data/{name}/records.jsonl').read_text().splitlines())
    oldids={r['id'] for r in old}; rows=[]
    for shard in sorted((ROOT/'data/fashion600/source/data').glob('*.parquet')):
        rows.extend(pq.read_table(shard,columns=['id','baseColour','articleType','gender','image']).to_pylist())
    rows=[r for r in rows if r['baseColour'] in {'Black','White','Blue','Red','Green','Grey','Brown','Pink'}
          and r['articleType'] in {'Tshirts','Shirts','Casual Shoes','Sports Shoes','Tops','Handbags'} and r['gender'] in {'Men','Women'}]
    rows.sort(key=lambda r:int(r['id']))
    parent=list(range(len(rows)))
    def find(i):
        while i!=parent[i]: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    # Four 16-bit buckets guarantee a shared bucket for Hamming distance <=3.
    # Five buckets guarantee one shared bucket at distance <=4.
    buckets=defaultdict(list); exact={}; phashes=[]; thumbnails=[]
    for i,r in enumerate(rows):
        im=Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')
        digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(); ph=dhash(im)
        r['digest']=digest; r['ph']=ph
        thumb=np.asarray(im.resize((32,32)),dtype=np.float32)
        if digest in exact: union(i,exact[digest])
        exact[digest]=i
        candidates=set()
        for b,(shift,bits) in enumerate(((0,13),(13,13),(26,13),(39,13),(52,12))):
            key=(b,(ph>>shift)&((1<<bits)-1)); candidates.update(buckets[key]); buckets[key].append(i)
        for j in candidates:
            if (ph^phashes[j]).bit_count()<=4 and np.abs(thumb-thumbnails[j]).mean()<=5:
                union(i,j)
        phashes.append(ph)
        thumbnails.append(thumb)
    components=defaultdict(list)
    for i in range(len(rows)): components[find(i)].append(i)
    rng=random.Random(SEED); groups=list(components.values()); rng.shuffle(groups)
    splitids=defaultdict(list)
    for group in groups:
        # ALL previously viewed examples/components are forced to training.
        split='train' if any(str(rows[i]['id']) in oldids for i in group) else rng.choices(['train','val','test'],[.6,.16,.24])[0]
        splitids[split].extend(group)
    limits={'train':8000,'val':1200,'test':1800}; out=ROOT/'data/fashion_expanded'; (out/'images').mkdir(parents=True,exist_ok=True)
    records=[]
    for split,ids in splitids.items():
        rng.shuffle(ids)
        for i in ids[:limits[split]]:
            r=rows[i]; relative=f"images/{r['id']}.jpg"; (out/relative).write_bytes(r['image']['bytes'])
            records.append({'id':str(r['id']),'image':relative,'split':split,
                'labels':dict(zip(meta['label_keys'],([r['baseColour']],[r['articleType']],[r['gender']]))),
                'image_sha256':r['digest'],'duplicate_component':find(i),'dhash':f"{r['ph']:016x}"})
    meta={k:meta[k] for k in ('dataset','revision','label_keys','directions','instruction')}
    meta.update(protocol='Decoded exact or (dHash<=4 AND 32x32 RGB mean absolute difference<=5/255) connected components split together; all pilot-containing components forced to train; capped at 8000/1200/1800. Seen attributes; no claim of unseen combination generalization.',
        components=len(components),largest_component=max(map(len,components.values())),
        candidate_count=len(rows),available_by_split={s:len(ids) for s,ids in splitids.items()},
        license='Original Kaggle listing MIT; image rights not independently cleared; no image redistribution.')
    save('fashion_expanded',records,meta)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('dataset',choices=['cub','dtd','fashion']); args=ap.parse_args()
    globals()[args.dataset]()
