"""Reuse immutable features only when image, basis, and encoder settings agree."""
import argparse
import json
from pathlib import Path
import shutil
import torch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--backend',choices=['omnijev','base','flair'],required=True); args=ap.parse_args()
    target=Path('runs/v3/fashion_expanded')/args.backend; target.mkdir(parents=True,exist_ok=True)
    sources=[Path('runs/fashion2400')/args.backend]
    if args.backend=='omnijev': sources.append(Path('runs/fashion600/features'))
    records={r['id']:r for r in (json.loads(s) for s in Path('data/fashion_expanded/records.jsonl').read_text().splitlines())}
    meta=json.loads(Path('data/fashion_expanded/metadata.json').read_text()); count=0
    for source in sources:
        if not (source/'provenance.json').exists(): continue
        prov=json.loads((source/'provenance.json').read_text())
        if prov['basis']!={'directions':meta['directions'],'instruction':meta['instruction']}: continue
        if prov.get('disable_lora',False)!=(args.backend=='base'): continue
        if prov.get('dataset_revision')!=meta['revision']: continue
        if (target/'provenance.json').exists():
            if json.loads((target/'provenance.json').read_text())!=prov: raise ValueError('Provenance mismatch')
        else: (target/'provenance.json').write_text(json.dumps(prov,indent=2))
        for file in source.glob('*.pt'):
            if file.stem not in records or (target/file.name).exists(): continue
            p=torch.load(file,map_location='cpu',weights_only=True)
            if p.get('image_sha256')!=records[file.stem]['image_sha256']: continue
            shutil.copy2(file,target/file.name); count+=1
    print(f'{args.backend}: reused {count} matching pilot feature files',flush=True)

if __name__=='__main__': main()
