"""Verify every v3 best head loads and matches its selected validation metrics."""
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding import OptionEmbeddingHead
from superembedding.research import evaluate,prepare_relations
from scripts.research_suite import load,project

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/results/v3'); grid=json.loads((root/'validation_grid.json').read_text())
    groups=defaultdict(list)
    for item in grid: groups[(item['dataset'],item['backend'])].append(item)
    checked=[]
    for (dataset,backend),items in groups.items():
        meta,records,prov,basis,u,zq,labels,splits,ident=load(Path('data')/dataset,Path('runs/v3')/dataset/backend)
        u,zq=u.cuda(),zq.cuda(); val=splits['val']
        relations=prepare_relations([labels[i] for i in val],meta['label_keys'],'cuda')
        for item in items:
            head,_,_=OptionEmbeddingHead.load(Path(item['path'])/'best_head.pt',basis); head=head.cuda().eval()
            slots,vectors=project(head,u,zq,val)
            metrics=evaluate(slots,vectors,meta['label_keys'],relations)
            expected=item['validation']['best_metrics']; errors=[]
            for task in ('attributes','pairs'):
                for name,details in expected[task].items():
                    for metric in ('mAP','R@1','R@5'):
                        if details.get(metric) is not None:
                            errors.append(abs(metrics[task][name][metric]-details[metric]))
            for metric in ('mAP','R@1','R@5'): errors.append(abs(metrics['joint'][metric]-expected['joint'][metric]))
            error=max(errors)
            if error>1e-6: raise AssertionError(f'Export metric mismatch {item["path"]}: {error}')
            assert torch.allclose(slots.norm(dim=-1),torch.ones_like(slots[:,:,0]),atol=1e-5)
            assert torch.allclose(vectors.norm(dim=-1),torch.ones_like(vectors[:,0]),atol=1e-5)
            checked.append({'path':item['path'],'max_validation_metric_difference':error,
                            'slots_shape':list(slots.shape),'embedding_shape':list(vectors.shape)})
            del head,slots,vectors
        print(f'Verified {dataset} {backend}: {len(items)} heads',flush=True)
        del u,zq,relations
    assert len(checked)==432
    (root/'export_verification.json').write_text(json.dumps({'count':len(checked),
        'maximum_metric_difference':max(x['max_validation_metric_difference'] for x in checked),'heads':checked},indent=2))

if __name__=='__main__': main()
