"""Seal the complete v4 validation grid, verify exports, then score fixed test sets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import OptionEmbeddingHead
from superembedding.baselines import DATASETS,MODELS
from superembedding.checkpoint import fingerprint,atomic_json
from superembedding.research import prepare_relations,evaluate,rank_metrics,relation
from scripts.research_suite import load,project

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def seal(root,out):
    allruns=[];selected=[];files=[]
    for d in DATASETS:
        for b in MODELS:
            for scale in (25,50,100):
                options={}
                for lr in (.0001,.001):
                    candidates=[]
                    for seed in (17,42,91):
                        path=root/d/'suite'/b/f'n{scale}'/f'raw_s{seed}_lr{lr:g}'
                        report=json.loads((path/'validation.json').read_text());cfg=report['config']
                        assert cfg['protocol']=='v4' and cfg['steps']==1000 and cfg['seed']==seed
                        assert cfg['backend']==b and cfg['lr']==lr and cfg['scale']==scale
                        assert [h['step'] for h in report['history']]==[250,500,750,1000]
                        last=torch.load(path/'last.pt',weights_only=True,map_location='cpu')
                        assert last['step']==1000 and last['fingerprint']==fingerprint(cfg)
                        assert last['extra']['best']==report['best']
                        for file in ('best_head.pt','last.pt'):
                            files.append({'path':(path/file).as_posix(),'sha256':sha(path/file),'bytes':(path/file).stat().st_size})
                        item=dict(dataset=d,backend=b,scale=scale,arch='raw',seed=seed,lr=lr,path=path.as_posix(),
                            head_sha256=sha(path/'best_head.pt'),validation=report)
                        candidates.append(item);allruns.append(item)
                    options[lr]=candidates
                best=max(options,key=lambda lr:sum(i['validation']['best'] for i in options[lr])/3)
                selected.extend(options[best])
    assert len(allruns)==324 and len(selected)==162 and len(files)==648
    result={'protocol':'v4','complete_runs':len(allruns),'selected_runs':len(selected),
        'selection_rule':'one LR per dataset/backend/scale by mean validation mAP across three seeds',
        'selected':selected,'selection_fingerprint':fingerprint(selected)}
    target=out/'selection_sealed.json'
    if target.exists() and json.loads(target.read_text())!=result:raise ValueError('Sealed selection changed')
    atomic_json(result,target);atomic_json(allruns,out/'validation_grid.json');atomic_json(files,out/'checkpoint_manifest.json')
    return result,allruns

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--root',default='runs/v4')
    ap.add_argument('--output',default='artifacts/results/v4');a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.output);root=Path(a.root);out.mkdir(parents=True,exist_ok=True)
    selection,allruns=seal(root,out);results=[];frozen=[];audit=[];provenances={}
    for d in DATASETS:
        for b in MODELS:
            meta,records,prov,basis,u,zq,labels,splits,ident=load(Path('data')/d,Path(a.features)/d/b)
            provenances[f'{d}/{b}']=prov;u,zq=u.cuda(),zq.cuda()
            valrel=prepare_relations([labels[i] for i in splits['val']],meta['label_keys'],'cpu')
            maxdiff=0.
            for item in [i for i in allruns if i['dataset']==d and i['backend']==b]:
                assert item['validation']['config']['data_fingerprint']==ident
                head,_,_=OptionEmbeddingHead.load(Path(item['path'])/'best_head.pt',basis);head=head.cuda()
                slots,vectors=project(head,u,zq,splits['val'])
                metrics=evaluate(slots,vectors,meta['label_keys'],valrel)
                diff=abs(metrics['attributes']['macro']['mAP']-item['validation']['best'])
                maxdiff=max(maxdiff,diff)
                assert diff<2e-6,(d,b,diff)
            audit.append(dict(dataset=d,backend=b,reloaded_heads=18,max_validation_mAP_difference=maxdiff))
            del valrel,head,slots,vectors
            test=splits['test'];rel=prepare_relations([labels[i] for i in test],meta['label_keys'],'cpu')
            extra=relation([[set(records[i]['joint_texture_labels'])] for i in test],(0,)) if d=='dtd' else None
            def score(slots,vectors):
                m=evaluate(slots,vectors,meta['label_keys'],rel)
                if extra is not None:m['joint_texture_overlap']=rank_metrics(vectors,*extra)
                return m
            for item in [i for i in selection['selected'] if i['dataset']==d and i['backend']==b]:
                path=Path(item['path']);assert sha(path/'best_head.pt')==item['head_sha256']
                target=out/'test_runs'/f"{d}_{b}_n{item['scale']}_s{item['seed']}.json"
                if target.exists():
                    r=json.loads(target.read_text());assert r['selection_fingerprint']==selection['selection_fingerprint']
                else:
                    head,_,_=OptionEmbeddingHead.load(path/'best_head.pt',basis);head=head.cuda()
                    slots,vectors=project(head,u,zq,test)
                    r={k:v for k,v in item.items() if k!='validation'}
                    r.update(metrics=score(slots,vectors),selection_fingerprint=selection['selection_fingerprint'],
                        best_step=item['validation']['best_step'],trainable_parameters=item['validation']['trainable_parameters'],
                        output_dimension=item['validation']['output_dimension'])
                    atomic_json(r,target)
                results.append(r)
            target=out/'test_runs'/f'{d}_{b}_frozen.json'
            if target.exists():
                r=json.loads(target.read_text());assert r['data_fingerprint']==ident
            else:
                slots=F.normalize(u[test],dim=-1);vectors=F.normalize(slots.flatten(1),dim=-1)
                r=dict(dataset=d,backend=b,data_fingerprint=ident,raw=score(slots,vectors),native_width=u.shape[-1])
                atomic_json(r,target)
            frozen.append(r)
            print(f'EVALUATED {d} {b}; export diff={maxdiff}',flush=True)
            del u,zq,rel
    atomic_json(dict(results=results,frozen_baselines=frozen,selection_fingerprint=selection['selection_fingerprint']),out/'test_results.json')
    atomic_json(audit,out/'export_verification.json');atomic_json(provenances,out/'feature_provenances.json')
    print(f'Finished {len(results)} selected tests; verified all {len(allruns)} exported heads',flush=True)

if __name__=='__main__':main()
