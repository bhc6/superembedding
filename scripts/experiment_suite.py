"""Train frozen-feature ablations with resumable state; never evaluate test here."""
import argparse
from dataclasses import asdict
import itertools
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import AttributeBasis, OptionEmbeddingHead, attribute_triplet_loss
from superembedding.checkpoint import fingerprint, save_training, restore_training, atomic_save, atomic_copy
from scripts.train_cached import retrieval_metrics

ARCHITECTURES = {
    'raw': ('raw', False, None, False),
    'gated_frozen': ('gated', False, None, False),
    'gated_tuned': ('gated', True, None, False),
    'residual': ('residual', True, None, False),
    'joint_concat': ('raw', False, None, True),
    'joint_out768': ('raw', False, 768, True),
    'joint_out256': ('raw', False, 256, True),
}


def load_data(data, cache):
    root, cache = Path(data), Path(cache)
    meta = json.loads((root/'metadata.json').read_text())
    records = [json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    provenance = json.loads((cache/'provenance.json').read_text())
    basis = AttributeBasis(tuple(meta['directions']), meta['instruction'])
    if AttributeBasis(**provenance['basis']) != basis:
        raise ValueError('Feature attribute basis mismatch')
    features=[]
    for r in records:
        p=torch.load(cache/f"{r['id']}.pt", weights_only=True)
        if p['id'] != r['id'] or p['image_sha256'] != r['image_sha256']:
            raise ValueError('Feature image mismatch')
        features.append(p)
    u=torch.stack([p['u'] for p in features]).float()
    zq=torch.stack([p['zq'] for p in features]).float()
    keys=meta['label_keys']
    maps={k:{v:i for i,v in enumerate(sorted({r['labels'][k] for r in records}))} for k in keys}
    labels=torch.tensor([[maps[k][r['labels'][k]] for k in keys] for r in records])
    splits={s:torch.tensor([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','val','test']}
    return meta,records,provenance,basis,u,zq,labels,splits


def triplet_pools(labels, indices, joint=False):
    ids=indices.tolist(); pools=[]
    for dims in ([tuple(range(labels.shape[1]))] if joint else [(k,) for k in range(labels.shape[1])]):
        pool=[]
        for a in ids:
            pos=[i for i in ids if i != a and all(labels[i,k]==labels[a,k] for k in dims)]
            neg=[i for i in ids if any(labels[i,k]!=labels[a,k] for k in dims)]
            if joint:
                hard=[i for i in neg if sum(int(labels[i,k]==labels[a,k]) for k in dims)==len(dims)-1]
                neg=hard or neg
            if pos and neg: pool.append((a,pos,neg))
        if not pool: raise ValueError('No triplets')
        pools.append(pool)
    return pools


def sample(pools, batch_size, rng):
    ids=[]; attrs=[]
    for _ in range(batch_size):
        k=rng.randrange(len(pools)); a,pos,neg=rng.choice(pools[k])
        ids.append((a,rng.choice(pos),rng.choice(neg))); attrs.append(k)
    return torch.tensor(ids),torch.tensor(attrs)


@torch.no_grad()
def project_all(head,u,zq,indices):
    head.eval(); slots=[]; vectors=[]
    for ids in indices.split(64):
        output=head(u[ids],zq[ids]); slots.append(output.slots.cpu()); vectors.append(output.embedding.cpu())
    return torch.cat(slots),torch.cat(vectors)


def evaluate(slots, vectors, labels, keys):
    result={'attributes':retrieval_metrics(slots,labels,keys)}
    pairs={}
    for dims in itertools.combinations(range(len(keys)),2):
        _,joint=torch.unique(labels[:,dims],dim=0,return_inverse=True)
        v=F.normalize(slots[:,dims,:].flatten(1),dim=-1)
        name='+'.join(keys[k] for k in dims)
        pairs[name]=retrieval_metrics(v[:,None],joint[:,None],[name])[name]
    result['pairs']=pairs
    _,joint=torch.unique(labels,dim=0,return_inverse=True)
    result['joint']=retrieval_metrics(vectors[:,None],joint[:,None],['joint'])['joint']
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/fashion2400')
    ap.add_argument('--features',default='runs/fashion2400/omnijev')
    ap.add_argument('--output',default='runs/fashion2400/suite')
    ap.add_argument('--architectures',nargs='+',default=list(ARCHITECTURES))
    ap.add_argument('--seeds',nargs='+',type=int,default=[17,42,91])
    ap.add_argument('--lrs',nargs='+',type=float,default=[1e-4,1e-3])
    ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--stop-after',type=int,default=0,help='Graceful interruption for resume verification')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    meta,records,prov,basis,u,zq,labels,splits=load_data(args.data,args.features)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    feature_digest=hashlib.sha256(u.numpy().tobytes()+zq.numpy().tobytes()).hexdigest()
    u,zq=u.to(device),zq.to(device)
    # Convert once to Python lists: avoid millions of scalar torch comparisons.
    pools=fast_pools(labels.tolist(),splits['train'].tolist(),False)
    joint_pools=fast_pools(labels.tolist(),splits['train'].tolist(),True)
    decision=None
    if any(ARCHITECTURES[a][0]!='raw' for a in args.architectures):
        sys.path.insert(0,str(Path('repos/OmniJev').resolve()))
        from mso.head import OptionScorer
        decision=OptionScorer(u.shape[-1])
        decision.load_state_dict(torch.load(Path(prov['adapter'])/'head.pt',map_location='cpu',weights_only=True))
    stable_prov={k:v for k,v in prov.items() if k not in {'adapter','base'}}
    if 'adapter' in prov:
        stable_prov['decision_head_sha256']=hashlib.file_digest((Path(prov['adapter'])/'head.pt').open('rb'),'sha256').hexdigest()
    data_id=fingerprint({'records':records,'provenance':stable_prov,'feature_sha256':feature_digest})
    for arch,seed,lr in itertools.product(args.architectures,args.seeds,args.lrs):
        mode,train_gate,outdim,joint_loss=ARCHITECTURES[arch]
        out=Path(args.output)/f'{arch}_s{seed}_lr{lr:g}'
        config={'arch':arch,'seed':seed,'lr':lr,'steps':args.steps,'batch_size':64,
                'data_fingerprint':data_id,'basis':asdict(basis),'margin':.2,
                'loss':'attribute+joint' if joint_loss else 'attribute','embed_dim':256,
                'feature_width':u.shape[-1],'checkpoint_version':1}
        out.mkdir(parents=True,exist_ok=True)
        if (out/'validation.json').exists():
            prior=json.loads((out/'validation.json').read_text())
            if prior['config']!=json.loads(json.dumps(config)): raise ValueError('Completed run config mismatch')
            print(f'SKIP {out.name}',flush=True); continue
        torch.manual_seed(seed); random.seed(seed); rng=random.Random(seed)
        kwargs=dict(num_slots=len(basis.directions),output_dim=outdim)
        if mode=='raw': head=OptionEmbeddingHead(u.shape[-1],embed_dim=256,mode=mode,**kwargs)
        else: head=OptionEmbeddingHead.from_decision_head(decision,256,mode,train_gate,**kwargs)
        head=head.to(device)
        opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
        source={'decision_checkpoint':prov.get('adapter','FLAIR'),'base_model':prov.get('base','FLAIR')}
        if prov.get('disable_lora'):
            source['feature_backend']='omnijev_lora_disabled'
        state={'best':-1.,'best_step':0,'history':[]}; step0=0
        def score_validation():
            slots,vectors=project_all(head,u,zq,splits['val'])
            metrics=evaluate(slots,vectors,labels[splits['val']],meta['label_keys'])
            score=metrics['attributes']['macro']['mAP']
            if joint_loss: score=(score+metrics['joint']['mAP'])/2
            return score,metrics
        if (out/'last.pt').exists():
            step0,state=restore_training(out/'last.pt',head,opt,config,rng)
            print(f'RESUME {out.name} step={step0}',flush=True)
        else:
            score,metrics=score_validation(); state.update(best=score,best_metrics=metrics)
            head.save(out/'best_head.pt',basis,source)
            save_training(out/'last.pt',head,opt,0,config,rng,state)
        start=time.perf_counter()
        for step in range(step0+1,args.steps+1):
            head.train()
            ids,attrs=sample(pools,64,rng); flat=ids.flatten()
            output=head(u[flat],zq[flat]); slots=output.slots.reshape(64,3,len(basis.directions),256)
            loss=attribute_triplet_loss(slots[:,0],slots[:,1],slots[:,2],attrs.to(device))
            if joint_loss:
                ids,_=sample(joint_pools,64,rng); flat=ids.flatten()
                v=head(u[flat],zq[flat]).embedding.reshape(64,3,-1)
                loss=loss+F.relu((v[:,0]*v[:,2]).sum(-1)-(v[:,0]*v[:,1]).sum(-1)+.2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient')
            opt.step()
            checkpoint=step%50==0 or step==args.steps or step==args.stop_after
            if step%50==0 or step==args.steps:
                score,metrics=score_validation()
                if score>state['best']:
                    state.update(best=score,best_step=step,best_metrics=metrics)
                    head.save(out/'best_head.pt',basis,source)
                state['history'].append({'step':step,'loss':loss.item(),'score':score})
                print(json.dumps({'run':out.name,**state['history'][-1]}),flush=True)
            if checkpoint:
                save_training(out/'last.pt',head,opt,step,config,rng,state)
                if step%250==0: atomic_copy(out/'last.pt',out/f'step_{step:06d}.pt')
            if step==args.stop_after:
                print('Stopped at completed optimizer update; rerun without --stop-after.',flush=True); return
        report={'config':config,**state,'trainable_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
                'elapsed_this_invocation':time.perf_counter()-start,'test_evaluated':False}
        (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        del head,opt


def fast_pools(labels,ids,joint):
    pools=[]
    dims_list=[tuple(range(len(labels[0])))] if joint else [(k,) for k in range(len(labels[0]))]
    for dims in dims_list:
        groups={}
        for i in ids: groups.setdefault(tuple(labels[i][k] for k in dims),[]).append(i)
        pool=[]
        for key,group in groups.items():
            neg=[i for i in ids if tuple(labels[i][k] for k in dims)!=key]
            if joint:
                hard=[i for i in neg if sum(labels[i][k]==key[j] for j,k in enumerate(dims))==len(dims)-1]
                neg=hard or neg
            if len(group)>1 and neg: pool.extend((a,[p for p in group if p!=a],neg) for a in group)
        if not pool: raise ValueError('No train triplets')
        pools.append(pool)
    return pools


if __name__=='__main__': main()
