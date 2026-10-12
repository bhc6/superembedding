"""Export only row identities/splits/hashes, never source images or annotations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

FIELDS=('id','source','source_row','source_revision','split','order','task',
        'original_rgb_sha256','image_file_sha256')


def public_record(record):
    return {key:record[key] for key in FIELDS}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    ap.add_argument('--output',default='artifacts/results/v5/rebuild');a=ap.parse_args()
    source=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    raw=(source/'records.jsonl').read_bytes();audit=json.loads((source/'audit.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=audit['manifest_sha256']:raise ValueError('Manifest audit mismatch')
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    public=[public_record(r) for r in rows]
    if len({r['id'] for r in public})!=len(public):raise ValueError('Duplicate source ID')
    payload=(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in public)).encode('utf-8')
    path=out/'rows.jsonl';temp=path.with_suffix('.jsonl.tmp');temp.write_bytes(payload);temp.replace(path)
    restored=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if restored!=public:raise ValueError('Rebuild identity round-trip failed')
    metadata={'schema':1,'source_repo':'HuggingFaceM4/the_cauldron',
        'source_revisions':sorted({r['source_revision'] for r in public}),
        'upstream_split':'train for every source; local split field is our experimental partition',
        'local_manifest_sha256':audit['manifest_sha256'],'public_manifest_sha256':hashlib.sha256(payload).hexdigest(),
        'records':len(public),'source_split_counts':dict(Counter(r['source']+'/'+r['split'] for r in public)),
        'fields':list(FIELDS),'contains_images':False,'contains_captions_questions_answers':False,
        'image_transform':'PIL RGB; thumbnail max side 640 LANCZOS; JPEG quality 90; compare saved file hash',
        'rebuild':'Fetch the pinned original source rows under their upstream terms, then apply listed identities/splits/order and verify hashes. Local file paths are deliberately excluded.',
        'limitations':'This is an identity manifest, not an annotation license grant or a complete independent download client.'}
    atomic_json(metadata,out/'metadata.json');print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
