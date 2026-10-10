"""Audit manifests and the complete validation grid before any v3 test scoring."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding import OptionEmbeddingHead,AttributeBasis
from superembedding.checkpoint import fingerprint
from scripts.research_suite import ARCHS

DATASETS=('fashion_expanded','cub_attributes','dtd')
SEEDS=(17,42,91)
LRS=(.0001,.001)

def architectures(backend,scale):
    if backend=='base': return ['raw']
    if backend=='flair': return ['raw'] if scale<100 else ['raw','joint_concat','joint_out_equal','joint_out_small']
    return list(ARCHS) if scale==100 else ['raw','joint_concat','joint_out_equal','joint_out_small']

def main():
    allruns=[]; selections=[]; datasets={}
    for dataset in DATASETS:
        data=Path('data')/dataset; meta=json.loads((data/'metadata.json').read_text())
        raw=(data/'records.jsonl').read_bytes(); rows=[json.loads(s) for s in raw.decode().splitlines()]
        assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256']
        assert len({r['id'] for r in rows})==len(rows)
        assert len({r['image_sha256'] for r in rows})==len(rows)
        components={}
        for r in rows:
            if 'duplicate_component' in r:
                key=r['duplicate_component']; assert components.setdefault(key,r['split'])==r['split']
        train=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['train_rank'])
        assert [r['train_rank'] for r in train]==list(range(len(train)))
        sizes=[meta['scales'][str(s)] for s in (25,50,100)]
        assert 0<sizes[0]<sizes[1]<sizes[2]==len(train)
        if dataset=='fashion_expanded':
            old=set()
            for name in ('fashion600','fashion2400'):
                old.update(json.loads(s)['image_sha256'] for s in (Path('data')/name/'records.jsonl').read_text().splitlines())
            assert not old&{r['image_sha256'] for r in rows if r['split']!='train'}
        basis=AttributeBasis(tuple(meta['directions']),meta['instruction'])
        extraction_audit={}
        for backend in ('omnijev','base','flair'):
            cache=Path('runs/v3')/dataset/backend
            extraction=json.loads((cache/'extraction.json').read_text())
            provenance=json.loads((cache/'provenance.json').read_text())
            assert len(list(cache.glob('*.pt')))==len(rows)
            if backend!='flair':
                assert provenance['encoding']=='same_layout_image_batch_v1' and provenance['batch_size']==4
                assert len(extraction['numerical_audit'])==16
                assert min(r['minimum_cosine_to_official'] for r in extraction['numerical_audit'])>=.999
            extraction_audit[backend]={'provenance':provenance,'extraction':extraction}
            for scale in (25,50,100):
                for arch in architectures(backend,scale):
                    grid={}
                    for lr in LRS:
                        runs=[]
                        for seed in SEEDS:
                            root=Path('runs/v3')/dataset/'suite'/backend/f'n{scale}'/f'{arch}_s{seed}_lr{lr:g}'
                            report=json.loads((root/'validation.json').read_text()); config=report['config']
                            assert config['steps']==1000 and config['train_count']==meta['scales'][str(scale)]
                            assert config['seed']==seed and config['lr']==lr and config['arch']==arch
                            assert config['backend']==backend and config['scale']==scale and not report['test_evaluated']
                            assert [x['step'] for x in report['history']]==[250,500,750,1000]
                            last=torch.load(root/'last.pt',weights_only=True,map_location='cpu')
                            assert last['step']==1000 and last['fingerprint']==fingerprint(config)
                            assert last['extra']['best']==report['best']
                            head,_,_=OptionEmbeddingHead.load(root/'best_head.pt',expected_basis=basis)
                            assert all(torch.isfinite(p).all() for p in head.parameters())
                            headhash=hashlib.file_digest((root/'best_head.pt').open('rb'),'sha256').hexdigest()
                            item={'dataset':dataset,'backend':backend,'scale':scale,'arch':arch,'seed':seed,'lr':lr,
                                  'path':root.as_posix(),'head_sha256':headhash,'validation':report}
                            runs.append(item); allruns.append(item)
                        grid[lr]=(sum(r['validation']['best'] for r in runs)/len(runs),runs)
                    selected_lr=max(LRS,key=lambda lr:grid[lr][0])
                    selections.extend(grid[selected_lr][1])
        datasets[dataset]={'count':len(rows),'splits':meta['splits'],'scales':meta['scales'],
                           'manifest_sha256':meta['manifest_sha256'],'labels_observed':meta['labels_observed'],
                           'feature_extraction':extraction_audit}
    assert len(allruns)==432 and len(selections)==216
    out=Path('artifacts/results/v3'); out.mkdir(parents=True,exist_ok=True)
    resume=json.loads((out/'resume_verification.json').read_text()); assert resume['model_max_abs_difference']==0
    payload={'protocol':'v3','datasets':datasets,'complete_runs':len(allruns),'selected_runs':len(selections),
             'selection_rule':'one learning rate per dataset/backend/scale/architecture by three-seed mean validation score',
             'test_evaluated':False,'selection_fingerprint':fingerprint(selections),'selected':selections}
    target=out/'selection_sealed.json'
    if target.exists() and json.loads(target.read_text())!=payload: raise ValueError('Sealed selection changed')
    target.write_text(json.dumps(payload,indent=2))
    (out/'validation_grid.json').write_text(json.dumps(allruns,indent=2))
    print(json.dumps({k:v for k,v in payload.items() if k!='selected'},indent=2))

if __name__=='__main__': main()
