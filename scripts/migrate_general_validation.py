"""Explicit validation-only migration, preserving weights/optimizer/RNG and lineage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.checkpoint import atomic_json,atomic_save,fingerprint
from scripts.train_general import make_config

def tensor_digest(value):
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            digest.update(str((str(x.dtype),tuple(x.shape))).encode())
            digest.update(x.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):digest.update(str(key).encode());visit(x[key])
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        else:digest.update(repr(x).encode())
    visit(value);return digest.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent-run',required=True);ap.add_argument('--parent-data',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--data',required=True);a=ap.parse_args()
    parent=Path(a.parent_run);data=Path(a.data);out=Path(a.output)
    if (out/'last.pt').exists():raise RuntimeError('Child checkpoint already exists; never overwrite migration')
    if json.loads((parent/'status.json').read_text())['state']!='paused':raise RuntimeError('Pause parent at completed optimizer boundary first')
    saved=torch.load(parent/'last.pt',map_location='cpu',weights_only=True);old=saved['config']
    if saved['extra']['best_step'] or saved['step']>=old['validate_every']:
        raise RuntimeError('This migration is restricted to before the first validation selection')
    source=Path(a.parent_data)/'records.jsonl';raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=old['data_sha256']:raise RuntimeError('Parent manifest changed')
    rows=[json.loads(s) for s in raw.decode('utf-8').splitlines()]
    audit=json.loads(Path('artifacts/results/v5/perceptual_overlap_audit.json').read_text())
    if audit['manifest_sha256']!=old['data_sha256']:raise RuntimeError('Near-duplicate audit stale')
    if audit['external_test_images_flagged']:
        raise RuntimeError('External test overlap candidates require separate adjudication before migration')
    flagged={r['held_id'] for r in audit['pairs'] if r['held_split'] in {'val','ood'}}
    kept=[r for r in rows if r['id'] not in flagged]
    train_before=[r for r in rows if r['split']=='train'];train_after=[r for r in kept if r['split']=='train']
    if train_before!=train_after:raise RuntimeError('Validation-only migration cannot change training rows or order')
    data.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    newraw=(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept)).encode('utf-8')
    (data/'records.jsonl').write_bytes(newraw)
    audit_data=json.loads((Path(a.parent_data)/'audit.json').read_text())
    audit_data.update(manifest_sha256=hashlib.sha256(newraw).hexdigest(),count=len(kept),
        split_counts=dict(Counter(r['split'] for r in kept)),
        source_split_counts=dict(Counter(r['source']+'/'+r['split'] for r in kept)),
        supervision_pairs=sum(len(r['texts']) for r in kept),
        perceptual_exclusions=sorted(flagged),
        exclusion_policy='Conservative held-only quarantine: pHash<=6 and dHash<=6. Includes template-similar false positives; not all confirmed duplicates.')
    atomic_json(audit_data,data/'audit.json');atomic_json(audit_data,'artifacts/results/v5/data_v51_audit.json')
    meta=json.loads((Path(a.parent_data)/'metadata.json').read_text());meta.update(records=len(kept),manifest_sha256=audit_data['manifest_sha256'])
    atomic_json(meta,data/'metadata.json')
    arg_keys=['steps','batch_size','lr','seed','max_pixels','checkpoint_every','validate_every','milestone_every']
    args=SimpleNamespace(**{k:old[k] for k in arg_keys},validation_limit=0,data=str(data),output=str(out),stop_after=0)
    new=make_config(args,newraw)
    allowed={'protocol','data_sha256','implementation_sha256','validation_limit'}
    if any(old.get(k)!=new.get(k) for k in set(old)|set(new) if k not in allowed):raise RuntimeError('Unexpected training configuration change')
    for k,h in old['implementation_sha256'].items():
        snapshot=parent/'implementation_snapshot'/k
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=h:raise RuntimeError('Missing original code snapshot')
        if k!='scripts/train_general.py' and new['implementation_sha256'][k]!=h:raise RuntimeError('Encoder/loss/checkpoint implementation changed')
    immutable_keys=['model','optimizer','rng_local','rng_python','rng_torch','rng_cuda','step']
    before=tensor_digest({k:saved[k] for k in immutable_keys})
    lineage={'parent_checkpoint_sha256':hashlib.sha256((parent/'last.pt').read_bytes()).hexdigest(),
        'parent_manifest_sha256':old['data_sha256'],'child_manifest_sha256':new['data_sha256'],
        'step':saved['step'],'reason':'Repair cross-source validation overlap before first selection; full validation replaces 16/source monitor',
        'training_records_identical':True,'training_records_sha256':fingerprint(train_before),
        'quarantined_held_records':len(flagged),'state_digest_before':before}
    saved['config']=new;saved['fingerprint']=fingerprint(new);saved['extra']['migration']=lineage
    atomic_save(saved,out/'last.pt');atomic_json(new,out/'config.json')
    reloaded=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
    after=tensor_digest({k:reloaded[k] for k in immutable_keys})
    if after!=before:raise RuntimeError('Migration changed weights/optimizer/RNG')
    lineage.update(state_digest_after=after,passed=True,
                   source_queue_preserved=saved['extra']['source_queue']==reloaded['extra']['source_queue'])
    atomic_json(lineage,out/'migration.json');atomic_json(lineage,'artifacts/results/v5/validation_migration.json')
    atomic_json({'state':'ready_to_resume','step':saved['step'],'last_checkpoint_step':saved['step']},out/'status.json')
    atomic_json({'run':str(out.resolve()),'data':str(data.resolve()),'protocol':new['protocol']},'runs/v5/active_general_run.json')
    print(json.dumps({**lineage,'splits':audit_data['split_counts']}),flush=True)

if __name__=='__main__':main()
