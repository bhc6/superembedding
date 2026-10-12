"""Bind current external data bytes and evaluator code before final evaluation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json,fingerprint


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def inventory():
    import pyarrow.parquet as pq
    store=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    science=store/'mmeb_scienceqa';urban=store/'urban1k';evidence=Path('data/evidence_diagnostic_v1')
    assets={}
    def add(label,path):assets[label]={'sha256':digest(path),'bytes':path.stat().st_size}
    for name in ['test.parquet','metadata.json']:add('scienceqa/'+name,science/name)
    metadata=json.loads((science/'metadata.json').read_text(encoding='utf-8'))
    if assets['scienceqa/test.parquet']['sha256']!=metadata['parquet_sha256']:raise ValueError('ScienceQA parquet differs from pinned download')
    rows=pq.read_table(science/'test.parquet').to_pylist()
    if len(rows)!=1000:raise ValueError('ScienceQA coverage differs')
    for rel in sorted({r['qry_img_path'] for r in rows}):add('scienceqa/'+rel,science/rel)
    add('urban1k/annotations/annotations.json',urban/'annotations/annotations.json')
    rows=json.loads((urban/'annotations/annotations.json').read_text(encoding='utf-8'))['annotations']
    if len(rows)!=1000 or sum(bool(r['caption'].strip()) for r in rows)!=999:raise ValueError('Urban1k coverage differs')
    for rel in sorted({r['image'] for r in rows}):add('urban1k/images/'+rel,urban/'images'/rel)
    for name in ['records.jsonl','metadata.json']:add('evidence/'+name,evidence/name)
    meta=json.loads((evidence/'metadata.json').read_text(encoding='utf-8'))
    if meta['records_sha256']!=assets['evidence/records.jsonl']['sha256']:raise ValueError('Evidence metadata changed')
    rows=[json.loads(s) for s in (evidence/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=256:raise ValueError('Evidence coverage differs')
    for rel in sorted({r[k] for r in rows for k in ['anchor','relevant_change','irrelevant_change']}):add('evidence/'+rel,evidence/rel)
    code=['scripts/evaluate_mmeb_scienceqa.py','scripts/evaluate_general_transfer.py','superembedding/general.py',
          'superembedding/omnijev.py','superembedding/heads.py','superembedding/general_export.py',
          'superembedding/general_inference.py']
    for path in code:add('implementation/'+path,Path(path))
    return {'protocol':'external-input-byte-seal-v1','assets':assets,'asset_count':len(assets),
            'historical_limit':'This seal establishes byte identity from its creation onward, not historical pretraining cleanliness or unchanged files before creation.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--phase',choices=['before','after','retrospective'],required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    current=inventory();path=out/'external_inputs_sealed.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['inventory']!=current:raise ValueError('External input/code byte identity changed; do not use old caches')
        if args.phase=='before' and old['phase_at_creation']!='before':raise ValueError('Retrospective seal cannot be relabeled prospective')
    else:
        if args.phase=='after':raise ValueError('No pre-evaluation seal exists')
        old={'inventory':current,'fingerprint':fingerprint(current),'phase_at_creation':args.phase,
             'created_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(old,path)
    result={'passed':True,'phase':args.phase,'fingerprint':fingerprint(current),'assets':current['asset_count'],
            'checked_utc':datetime.now(timezone.utc).isoformat(),'original_seal_phase':old['phase_at_creation']}
    atomic_json(result,out/f'external_inputs_{args.phase}_audit.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
