"""Read-only near-duplicate screening; no edits to an active training manifest."""
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

def main():
    root=Path('C:/Users/123/.cache/superembedding/v5/data')
    payload=(root/'records.jsonl').read_bytes()
    rows=[json.loads(s) for s in payload.decode('utf-8').splitlines()]
    # Include external evaluation images in the same train-vs-held screening.
    external=[]
    for name,folder in [('scienceqa','C:/Users/123/.cache/superembedding/v5/mmeb_scienceqa/ScienceQA'),
                        ('urban1k','C:/Users/123/.cache/superembedding/v5/urban1k/images')]:
        for path in sorted(Path(folder).glob('*.jpg')):
            with Image.open(path) as im:
                im=im.convert('RGB');rgb=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                im.thumbnail((640,640),Image.Resampling.LANCZOS)
                small=np.asarray(im.convert('L').resize((9,8)))
                dhash=''.join('1' if v else '0' for v in (small[:,1:]>small[:,:-1]).flatten())
            external.append({'id':f'{name}_official:{path.name}','split':'external_test','source':name,
                'image':str(path),'dhash':dhash,'original_rgb_sha256':rgb,
                'image_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(external)!=2000:raise RuntimeError('Require all 2000 external benchmark images')
    rows+=external
    # DCT-II low-frequency perceptual hash, exclude DC from threshold.
    x=np.arange(32);freq=np.arange(8)[:,None]
    basis=np.cos(np.pi*(x+.5)*freq/32)
    def features(row):
        with Image.open(row['image']) as im:
            gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
            small=np.asarray(im.convert('RGB').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)/255
        low=(basis@gray@basis.T).flatten()[1:]
        bits=low>np.median(low)
        value=sum(int(b)<<i for i,b in enumerate(bits))
        return value,small
    with ThreadPoolExecutor(4) as pool:feats=list(pool.map(features,rows))
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    held=[i for i,r in enumerate(rows) if r['split']!='train']
    th=np.array([feats[i][0] for i in train],dtype=np.uint64)
    td=np.array([int(rows[i]['dhash'],2) for i in train],dtype=np.uint64)
    matches=[]
    for i in held:
        pd=np.bitwise_count(th ^ np.uint64(feats[i][0]))
        dd=np.bitwise_count(td ^ np.uint64(int(rows[i]['dhash'],2)))
        # Screen using two hashes; do not equate distance with duplicate proof.
        candidates=np.where((pd<=6)&(dd<=6))[0]
        for offset in candidates:
            j=train[int(offset)]
            mse=float(np.mean((feats[i][1]-feats[j][1])**2))
            matches.append({'held_id':rows[i]['id'],'held_split':rows[i]['split'],'train_id':rows[j]['id'],
                'phash_distance':int(pd[offset]),'dhash_distance':int(dd[offset]),'rgb32_mse':mse,
                'same_original_rgb':rows[i]['original_rgb_sha256']==rows[j]['original_rgb_sha256'],
                'same_encoded_sha256':rows[i]['image_file_sha256']==rows[j]['image_file_sha256']})
    matches.sort(key=lambda r:(r['rgb32_mse'],r['phash_distance']+r['dhash_distance']))
    monitored={r['id'] for source in sorted({r['source'] for r in rows if r['split']=='train'})
        for r in sorted([r for r in rows if r['split']=='val' and r['source']==source],key=lambda r:r['id'])[:16]}
    result={'manifest_sha256':hashlib.sha256(payload).hexdigest(),'mode':'read-only screening',
        'train_images':len(train),'held_images':len(held),'candidate_pairs':len(matches),
        'held_images_flagged':len({r['held_id'] for r in matches}),
        'external_test_images_flagged':sorted({r['held_id'] for r in matches if r['held_split']=='external_test'}),
        'checkpoint_monitor_images_flagged':sorted({r['held_id'] for r in matches if r['held_id'] in monitored}),
        'confirmed_exact_pairs':sum(r['same_original_rgb'] or r['same_encoded_sha256'] for r in matches),
        'method':'63-bit DCT pHash Hamming<=6 AND existing 64-bit dHash Hamming<=6; RGB32 MSE for review order',
        'limit':'Screening only. Similar layouts, charts, synthetic objects can be false positives. Manual/semantic review required.',
        'pairs':matches}
    atomic_json(result,'artifacts/results/v5/perceptual_overlap_audit.json')
    # Contact sheet is local-only; source images are not redistribution assets.
    from PIL import ImageDraw
    lookup={r['id']:r for r in rows}
    top=matches[:20];sheet=Image.new('RGB',(700,max(1,len(top))*165),'white');draw=ImageDraw.Draw(sheet)
    for k,r in enumerate(top):
        for col,key in enumerate(['held_id','train_id']):
            with Image.open(lookup[r[key]]['image']) as im:
                im=im.convert('RGB');im.thumbnail((320,130));sheet.paste(im,(col*350,k*165+25))
            draw.text((col*350+3,k*165+3),f"{r[key]} p={r['phash_distance']} d={r['dhash_distance']} mse={r['rgb32_mse']:.5f}",fill='black')
    Path('runs/v5/data_audit').mkdir(parents=True,exist_ok=True)
    sheet.save('runs/v5/data_audit/near_duplicates.png')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)

if __name__=='__main__':main()
