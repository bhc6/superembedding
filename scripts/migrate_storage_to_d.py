"""Copy and SHA256-verify approved cache roots. Source removal is a separate step."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

MAPPINGS=(
    ('C:/Users/123/.cache/superembedding','D:/DEMO/superembedding-storage/superembedding'),
    ('C:/Users/123/.cache/huggingface','D:/DEMO/superembedding-storage/huggingface'),
    ('C:/Users/123/AppData/Local/pip/Cache','D:/DEMO/superembedding-storage/pip-cache'),
    ('C:/Users/123/AppData/Local/uv/cache','D:/DEMO/superembedding-storage/uv-cache'),
)


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def inventory(root):
    files={}
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in dirs+names:
            path=Path(folder)/name
            if path.is_symlink() or path.is_junction():raise ValueError('Unexpected nested filesystem link')
        for name in names:
            path=Path(folder)/name;stat=path.stat()
            files[path.relative_to(root).as_posix()]=(stat.st_size,stat.st_mtime_ns)
    return files


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--only',choices=[Path(dst).name for _,dst in MAPPINGS]);a=ap.parse_args()
    store=Path('D:/DEMO/superembedding-storage');store.mkdir(parents=True,exist_ok=True)
    summary=[]
    for source,destination in MAPPINGS:
        if a.only and Path(destination).name!=a.only:continue
        src=Path(source);dst=Path(destination)
        if src.is_junction():raise ValueError('Source already redirected; use saved migration audit')
        if not src.exists():continue
        initial=inventory(src);total=sum(v[0] for v in initial.values())
        if not a.verify_only and shutil.disk_usage(store).free<total+2*2**30:raise RuntimeError('Insufficient destination space')
        dst.mkdir(parents=True,exist_ok=True)
        def copy_one(name):
            origin=src/name;target=dst/name;target.parent.mkdir(parents=True,exist_ok=True)
            if a.verify_only:
                if not target.exists():raise ValueError('Missing copied file')
            elif not target.exists() or target.stat().st_size!=initial[name][0] or digest(origin)!=digest(target):
                temp=target.with_name(target.name+'.migration-tmp')
                shutil.copy2(origin,temp);os.replace(temp,target)
            before=digest(origin);after=digest(target)
            if before!=after:raise ValueError('Copied file checksum mismatch')
            return {'relative_path':name,'bytes':initial[name][0],'sha256':before}
        manifest=[]
        with ThreadPoolExecutor(4) as pool:
            for i,r in enumerate(pool.map(copy_one,sorted(initial)),1):
                manifest.append(r)
                if i%5000==0:print(json.dumps({'source':source,'verified_files':i,'total_files':len(initial)}),flush=True)
        if inventory(src)!=initial:raise RuntimeError('Source changed while migrating; do not remove source')
        if set(inventory(dst))!=set(initial):raise RuntimeError('Unexpected destination contents')
        # Private cache manifests stay outside the repository (may include credential file digests).
        label=dst.name
        atomic_json({'source':source,'destination':destination,'files':manifest},store/(label+'-migration-files.json'))
        record={'source':source,'destination':destination,'files':len(initial),'bytes':total,'all_sha256_match':True,
                'source_stable':True,'verified_utc':datetime.now(timezone.utc).isoformat()}
        summary.append(record);atomic_json({'state':'verified_partial','roots':summary},store/'migration_verified.json')
        print(json.dumps(record),flush=True)
    atomic_json({'state':'verified','roots':summary,'total_bytes':sum(r['bytes'] for r in summary)},store/'migration_verified.json')


if __name__=='__main__':main()
