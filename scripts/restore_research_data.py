"""Reconstruct the published frozen manifests from official sources, without pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
DATASETS=('fashion_expanded','cub_attributes','dtd')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    if not args.verify_only:
        subprocess.run([sys.executable,'scripts/download_research_data.py'],cwd=ROOT,check=True)
    rows_by_dataset={}
    for name in DATASETS:
        source=ROOT/'artifacts/data'/name; target=ROOT/'data'/name
        if not args.verify_only: target.mkdir(parents=True,exist_ok=True)
        for filename in ('metadata.json','records.jsonl'):
            if (target/filename).exists():
                if (target/filename).read_text(encoding='utf-8')!=(source/filename).read_text(encoding='utf-8'):
                    raise ValueError(f'Existing manifest differs from release: {target/filename}')
            elif not args.verify_only: (target/filename).write_bytes((source/filename).read_bytes())
            else: raise FileNotFoundError(target/filename)
        rows_by_dataset[name]=[json.loads(s) for s in (target/'records.jsonl').read_text().splitlines()]
    if not args.verify_only:
        fashion=ROOT/'data/fashion_expanded';(fashion/'images').mkdir(exist_ok=True)
        missing={r['id']:r for r in rows_by_dataset['fashion_expanded'] if not (fashion/r['image']).exists()}
        if missing:
            for shard in ('train-00000-of-00002-6cff4c59f91661c3.parquet','train-00001-of-00002-bb459e5ac5f01e71.parquet'):
                path=ROOT/'data/fashion600/source/data'/shard
                if not path.exists():
                    path=Path(hf_hub_download('ashraq/fashion-product-images-small','data/'+shard,repo_type='dataset',
                        revision='3859c76db2f6f3d3b9a3863345e3ccdbff75879d',local_dir=ROOT/'data/fashion_expanded/source'))
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['id','image']):
                    for row in batch.to_pylist():
                        ident=str(row['id'])
                        if ident in missing:
                            (fashion/missing.pop(ident)['image']).write_bytes(row['image']['bytes'])
            if missing: raise ValueError('Some release IDs are missing from pinned Fashion source')
    result={}
    for name,rows in rows_by_dataset.items():
        root=ROOT/'data'/name
        for r in rows:
            with Image.open(root/r['image']) as im:
                im=im.convert('RGB');digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            if digest!=r['image_sha256']: raise ValueError(f'Image checksum mismatch: {name}/{r["id"]}')
        result[name]={'verified_images':len(rows),'all_decoded_image_hashes_match':True}
        print(json.dumps({name:result[name]}),flush=True)
    target=ROOT/'artifacts/results/v3/image_verification.json'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
