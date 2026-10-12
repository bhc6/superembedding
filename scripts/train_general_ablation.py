"""CPU head-only grid; validation sealing precedes any OOD evaluation."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding.general_ablation import ARCHITECTURES,GeneralAblationHead
from superembedding.general import contrastive_loss,positive_mask
from superembedding.checkpoint import atomic_json,fingerprint,save_training,restore_training


def read_bank(root, splits):
    root=Path(root);prov=json.loads((root/'provenance.json').read_text());sig=fingerprint(prov)
    status=json.loads((root/'status.json').read_text())
    if status.get('state')!='completed' or status.get('fingerprint')!=sig:raise ValueError('Feature extraction incomplete')
    records=[]
    for i,expected in enumerate(prov['ids']):
        r=torch.load(root/'shards'/f'{i:05d}.pt',weights_only=True,map_location='cpu')
        if r['fingerprint']!=sig or r['id']!=expected:raise ValueError('Feature identity mismatch')
        if r['split'] in splits:records.append(r)
    q=torch.stack([r['q'] for r in records]);d=torch.stack([r['d'] for r in records])
    digest=hashlib.sha256(q.numpy().tobytes()+d.numpy().tobytes()).hexdigest()
    return prov,records,q,d,digest


@torch.no_grad()
def evaluate(head, q, d, rows, indices):
    result={}
    for source in sorted({rows[i]['source'] for i in indices}):
        ids=[i for i in indices if rows[i]['source']==source]
        if head is None:
            qv=F.normalize(F.normalize(q[ids],dim=-1).flatten(1),dim=-1)
            dv=F.normalize(F.normalize(d[ids],dim=-1).flatten(1),dim=-1)
        else:
            qv=torch.cat([head(q[b]) for b in torch.tensor(ids).split(64)])
            dv=torch.cat([head(d[b]) for b in torch.tensor(ids).split(64)])
        mask=positive_mask([rows[i]['key'] for i in ids],'cpu')
        scores=qv@dv.T
        # Candidates with equal scores use an ID hash, independent of relevance.
        order=sorted(range(len(ids)),key=lambda j:hashlib.sha256(rows[ids[j]]['id'].encode()).hexdigest())
        ranks=torch.tensor(order)[scores[:,order].argsort(dim=-1,descending=True,stable=True)]
        hits=mask.gather(1,ranks)
        rr=(hits.float()/torch.arange(1,len(ids)+1)).max(-1).values
        result[source]={'queries':len(ids),'R1':hits[:,0].float().mean().item(),'MRR':rr.mean().item(),
                        'mean_positive_candidates':mask.sum(-1).float().mean().item()}
    return {'macro_r1':sum(v['R1'] for v in result.values())/len(result),'per_source':result}


def run_one(bank, output, arch, seed, lr, steps, stop_after=0):
    prov,rows,q,d,digest=bank;out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config={'protocol':'general-frozen-head-ablation-v1','architecture':arch,'seed':seed,'lr':lr,
        'steps':steps,'batch_size':32,'validation_every':100,'checkpoint_every':100,
        'feature_provenance':fingerprint(prov),'feature_tensor_sha256':digest,
        'loss':'same-source multi-positive InfoNCE tau .05','optimizer':'AdamW wd .01 clip 1',
        'implementation_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['scripts/train_general_ablation.py','superembedding/general_ablation.py','superembedding/general.py','superembedding/checkpoint.py']}}
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text())
        if result['config']!=config:raise ValueError('Completed ablation configuration mismatch')
        return result
    torch.manual_seed(seed);random.seed(seed);rng=random.Random(seed)
    head=GeneralAblationHead(arch);optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
    train={s:[i for i,r in enumerate(rows) if r['split']=='train' and r['source']==s]
           for s in sorted({r['source'] for r in rows if r['split']=='train'})}
    val=[i for i,r in enumerate(rows) if r['split']=='val']
    extra={'best_score':-1.,'best_step':0,'source_queue':[],'history':[]};start=0
    if (out/'last.pt').exists():start,extra=restore_training(out/'last.pt',head,optimizer,config,rng)
    def save(name,step):
        if shutil.disk_usage(out).free<700*2**20:raise RuntimeError('Insufficient disk reserve for ablation checkpoint')
        save_training(out/name,head,optimizer,step,config,rng,extra)
    def validate(step):
        score=evaluate(head,q,d,rows,val);extra['history'].append({'step':step,'validation':score})
        if score['macro_r1']>extra['best_score']:
            extra['best_score']=score['macro_r1'];extra['best_step']=step;save('best.pt',step)
        return score
    if start==0 and not (out/'last.pt').exists():validate(0);save('last.pt',0)
    for step in range(start+1,steps+1):
        if not extra['source_queue']:
            extra['source_queue']=list(train);rng.shuffle(extra['source_queue'])
        source=extra['source_queue'].pop()
        for _ in range(32):
            ids=rng.sample(train[source],32);keys=[rows[i]['key'] for i in ids]
            if not positive_mask(keys,'cpu').all():break
        else:raise ValueError('No negatives after resampling')
        optimizer.zero_grad(set_to_none=True)
        loss=contrastive_loss(head(q[ids]),head(d[ids]),keys)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(norm) or not torch.isfinite(loss):raise RuntimeError('Nonfinite ablation update')
        optimizer.step()
        if step%100==0 or step==steps:validate(step)
        if step%100==0 or step in {steps,stop_after}:
            save('last.pt',step)
            atomic_json({'state':'running','step':step,'best_step':extra['best_step'],'best_validation':extra['best_score']},out/'status.json')
        if step==stop_after:
            atomic_json({'state':'paused','step':step},out/'status.json');return None
    result={'config':config,'best_step':extra['best_step'],'best_validation':extra['best_score'],
        'history':extra['history'],'parameters':sum(p.numel() for p in head.parameters()),
        'output_dimension':head(q[:1]).shape[-1],'step_zero_selection':extra['best_step']==0}
    atomic_json(result,out/'result.json');atomic_json({'state':'completed','step':steps},out/'status.json')
    print(json.dumps({'architecture':arch,'seed':seed,'lr':lr,'best_step':extra['best_step'],'validation':extra['best_score']}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():raise RuntimeError('CPU ablation expected')
    bank=read_bank(args.features,{'train','val'});out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    seeds=[17,42,91];lrs=[1e-4,1e-3];results=[]
    for arch,seed,lr in itertools.product(ARCHITECTURES,seeds,lrs):
        run=out/f'{arch}_s{seed}_lr{lr:g}'
        results.append({'run':str(run),**run_one(bank,run,arch,seed,lr,1000)})
    selection=[]
    for arch in ARCHITECTURES:
        means={lr:sum(r['best_validation'] for r in results if r['config']['architecture']==arch and r['config']['lr']==lr)/len(seeds) for lr in lrs}
        selected=max(lrs,key=lambda lr:(means[lr],-lr))
        selection.append({'architecture':arch,'learning_rate':selected,'mean_validation':means[selected],
            'runs':[r['run'] for r in results if r['config']['architecture']==arch and r['config']['lr']==selected]})
    sealed={'protocol':'general-frozen-head-ablation-v1','feature_provenance':fingerprint(bank[0]),
        'selection_rule':'one lr per architecture by three-seed mean validation R1; ties prefer lower lr',
        'results':results,'selected':selection,'ood_used_for_selection':False}
    atomic_json(sealed,out/'selection_sealed.json')
    atomic_json(sealed,'artifacts/results/v5/general_ablation_selection.json')
    atomic_json({'state':'completed','runs':len(results),'selected':len(selection)*len(seeds)},out/'status.json')


if __name__=='__main__':
    from filelock import FileLock
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',required=True)
    args,_=parser.parse_known_args();Path(args.output).mkdir(parents=True,exist_ok=True)
    with FileLock(str(Path(args.output)/'training.lock'),timeout=0):main()
