"""Evaluate only validation-sealed selections; never use test for model selection."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import OptionEmbeddingHead
from superembedding.checkpoint import fingerprint
from superembedding.research import prepare_relations,evaluate,relation,rank_metrics,random_ranking_expectation
from scripts.research_suite import load,project

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/results/v3'); seal=json.loads((root/'selection_sealed.json').read_text())
    assert seal['selection_fingerprint']==fingerprint(seal['selected'])
    grouped=defaultdict(list)
    for item in seal['selected']: grouped[(item['dataset'],item['backend'])].append(item)
    complete=[]; frozen=[]; chance={}
    for (dataset,backend),items in grouped.items():
        meta,records,prov,basis,u,zq,labels,splits,ident=load(Path('data')/dataset,Path('runs/v3')/dataset/backend)
        u,zq=u.cuda(),zq.cuda(); test=splits['test']
        relations=prepare_relations([labels[i] for i in test],meta['label_keys'],'cpu')
        if dataset not in chance:
            attributes={key:random_ranking_expectation(*relations[(i,)]) for i,key in enumerate(meta['label_keys'])}
            attributes['macro']={metric:sum(v[metric] for v in attributes.values())/len(attributes) for metric in ('mAP','R@1','R@5')}
            chance[dataset]={'attributes':attributes,'joint':random_ranking_expectation(*relations[tuple(range(len(meta['label_keys'])))])}
        extra=None
        if dataset=='dtd': extra=relation([[set(records[i]['joint_texture_labels'])] for i in test],(0,))
        def metrics(slots,vectors):
            result=evaluate(slots,vectors,meta['label_keys'],relations)
            if extra is not None: result['joint_texture_overlap']=rank_metrics(vectors,*extra)
            return result
        for item in items:
            path=Path(item['path']); name=f"{dataset}_{backend}_n{item['scale']}_{item['arch']}_s{item['seed']}"
            target=root/'test_runs'/f'{name}.json'; target.parent.mkdir(exist_ok=True)
            if hashlib.file_digest((path/'best_head.pt').open('rb'),'sha256').hexdigest()!=item['head_sha256']:
                raise ValueError('Selected head changed after sealing')
            if item['validation']['config']['data_fingerprint']!=ident: raise ValueError('Data/features changed after validation')
            if target.exists():
                result=json.loads(target.read_text())
                if result['selection_fingerprint']!=seal['selection_fingerprint']: raise ValueError('Test cache selection mismatch')
            else:
                head,_,_=OptionEmbeddingHead.load(path/'best_head.pt',basis); head=head.cuda()
                slots,vectors=project(head,u,zq,test)
                result={k:v for k,v in item.items() if k!='validation'}
                result.update(metrics=metrics(slots,vectors),validation_score=item['validation']['best'],
                    best_step=item['validation']['best_step'],output_dimension=item['validation']['output_dimension'],
                    trainable_parameters=item['validation']['trainable_parameters'],
                    selection_fingerprint=seal['selection_fingerprint'])
                temp=target.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)); temp.replace(target)
                del head,slots,vectors
            complete.append(result); print(f'EVALUATED {name}',flush=True)
        target=root/'test_runs'/f'{dataset}_{backend}_frozen.json'
        if target.exists():
            result=json.loads(target.read_text())
            if result['data_fingerprint']!=ident: raise ValueError('Frozen test cache mismatch')
        else:
            slots=F.normalize(u[test],dim=-1); vectors=F.normalize(slots.flatten(1),dim=-1)
            result={'dataset':dataset,'backend':backend,'data_fingerprint':ident,'raw':metrics(slots,vectors)}
            if backend=='flair':
                global_v=F.normalize(zq[test],dim=-1)
                result['global']=metrics(global_v[:,None].expand(-1,len(meta['label_keys']),-1),global_v)
            target.write_text(json.dumps(result,indent=2))
        frozen.append(result)
        del u,zq,relations
    (root/'test_results.json').write_text(json.dumps({'selection_fingerprint':seal['selection_fingerprint'],
        'results':complete,'frozen_baselines':frozen,'random_ranking_expectation':chance},indent=2))
    print(f'Finished {len(complete)} preselected model evaluations; no selection uses test.',flush=True)

if __name__=='__main__': main()
