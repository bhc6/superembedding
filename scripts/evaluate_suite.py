"""Select LR using validation only, then evaluate all three seeds on held-out test."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import OptionEmbeddingHead
from scripts.experiment_suite import load_data,project_all,evaluate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default='runs/fashion2400/omnijev')
    ap.add_argument('--suite',default='runs/fashion2400/suite')
    ap.add_argument('--output',default='runs/fashion2400/results_omnijev.json')
    args=ap.parse_args()
    if Path(args.output).exists():
        print('Existing test report retained; use a new output path only for an explicitly distinct evaluation.')
        return
    torch.set_num_threads(4)
    meta,records,prov,basis,u,zq,labels,splits=load_data('data/fashion2400',args.features)
    runs={}
    for path in Path(args.suite).glob('*/validation.json'):
        report=json.loads(path.read_text()); c=report['config']
        runs.setdefault(c['arch'],{}).setdefault(c['lr'],[]).append((path.parent,report))
    selections={}
    for arch,lrs in runs.items():
        if set(lrs)!={.0001,.001} or any(r['config']['steps']!=500 for group in lrs.values() for _,r in group):
            raise ValueError('Incomplete learning-rate grid or wrong training budget')
        if any(sorted(r['config']['seed'] for _,r in group)!=[17,42,91] for group in lrs.values()):
            raise ValueError(f'Incomplete seed set for {arch}')
        best_lr=max(lrs,key=lambda lr:statistics.mean(r['best'] for _,r in lrs[lr]))
        selections[arch]={'lr':best_lr,'validation_grid':{str(lr):statistics.mean(r['best'] for _,r in group) for lr,group in lrs.items()},'seeds':[]}
    # Freeze choices before opening test metrics; manifest available for audit.
    selection_path=Path(args.output).with_suffix('.selection.json')
    selection_path.write_text(json.dumps(selections,indent=2))
    u,zq=u.cuda(),zq.cuda()
    test=splits['test']; y=labels[test]
    slots=F.normalize(u[test].cpu(),dim=-1)
    baseline=evaluate(slots,F.normalize(slots.flatten(1),dim=-1),y,meta['label_keys'])
    # FLAIR also exposes a global embedding (same representation for every attribute).
    global_baseline=None
    if 'flair' in args.features.lower():
        v=F.normalize(zq[test].cpu(),dim=-1)
        global_baseline=evaluate(v[:,None].expand(-1,3,-1),v,y,meta['label_keys'])
    for arch,selection in selections.items():
        for folder,r in runs[arch][selection['lr']]:
            head,_,_=OptionEmbeddingHead.load(folder/'best_head.pt',basis)
            head=head.cuda()
            s,v=project_all(head,u,zq,test)
            metrics=evaluate(s,v,y,meta['label_keys'])
            selection['seeds'].append({'seed':r['config']['seed'],'best_step':r['best_step'],
                'checkpoint':str(folder/'best_head.pt'),'trainable_parameters':r['trainable_parameters'],
                'global_dim':v.shape[-1],'test':metrics})
        summary={}
        for key,extract in [('attribute_mAP',lambda r:r['test']['attributes']['macro']['mAP']),
                            ('attribute_R1',lambda r:r['test']['attributes']['macro']['R@1']),
                            ('joint_mAP',lambda r:r['test']['joint']['mAP']),
                            ('joint_R1',lambda r:r['test']['joint']['R@1']),
                            ('pair_mAP',lambda r:statistics.mean(p['mAP'] for p in r['test']['pairs'].values()))]:
            values=[extract(r) for r in selection['seeds']]
            summary[key]={'mean':statistics.mean(values),'std':statistics.stdev(values)}
        selection['summary']=summary
    report={'data':meta,'features':prov,'selection_rule':'mean validation score across three seeds; test not used for LR/checkpoint selection',
            'frozen_features':baseline,'global_features':global_baseline,'architectures':selections}
    Path(args.output).write_text(json.dumps(report,indent=2))
    for arch,s in selections.items(): print(arch,json.dumps(s['summary']),flush=True)


if __name__=='__main__': main()
