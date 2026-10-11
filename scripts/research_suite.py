"""Multi-domain, nested-data experiments with atomic exact-resume checkpoints.

Training uses train labels only; selection uses validation; test is a separate command.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import AttributeBasis,OptionEmbeddingHead,attribute_triplet_loss
from superembedding.checkpoint import fingerprint,save_training,restore_training,atomic_json
from superembedding.research import label_sets,ObservedTriplets,prepare_relations,evaluate

ARCHS={
    'raw':('raw',False,None,False),
    'gated_frozen':('gated',False,None,False),
    'gated_tuned':('gated',True,None,False),
    'residual':('residual',True,None,False),
    'joint_concat':('raw',False,None,True),
    'joint_out_equal':('raw',False,'equal',True),
    'joint_out_small':('raw',False,'small',True),
}

def load(data,cache):
    data,cache=Path(data),Path(cache)
    meta=json.loads((data/'metadata.json').read_text())
    records=[json.loads(s) for s in (data/'records.jsonl').read_text().splitlines()]
    prov=json.loads((cache/'provenance.json').read_text())
    basis=AttributeBasis(tuple(meta['directions']),meta['instruction'])
    if AttributeBasis(**prov['basis'])!=basis: raise ValueError('Feature basis mismatch')
    if (cache/'features.pt').exists():
        from superembedding.baselines import load_packed
        u,zq=load_packed(cache,records,prov)
    else:
        features=[]
        for r in records:
            f=torch.load(cache/f"{r['id']}.pt",weights_only=True,map_location='cpu')
            if f['id']!=r['id'] or f['image_sha256']!=r['image_sha256']: raise ValueError('Feature record mismatch')
            features.append(f)
        u=torch.stack([f['u'] for f in features]).float(); zq=torch.stack([f['zq'] for f in features]).float()
    digest=hashlib.sha256(u.numpy().tobytes()+zq.numpy().tobytes()).hexdigest()
    stable_prov={k:v for k,v in prov.items() if k not in ('adapter','base')}
    if 'adapter' in prov:
        adapter=Path(prov['adapter'])
        stable_prov['decision_head_sha256']=hashlib.file_digest((adapter/'head.pt').open('rb'),'sha256').hexdigest()
        for key in ('adapter','base'):
            ref=Path(prov[key])/'source_revision.json'
            if not ref.exists(): raise FileNotFoundError(f'Pinned model identity required: {ref}')
            stable_prov[key+'_identity']=json.loads(ref.read_text())
    ident=fingerprint({'records':records,'provenance':stable_prov,'features_sha256':digest})
    splits={s:[i for i,r in enumerate(records) if r['split']==s] for s in ('train','val','test')}
    return meta,records,prov,basis,u,zq,label_sets(records,meta['label_keys']),splits,ident

@torch.no_grad()
def project(head,u,zq,indices):
    head.eval(); outputs=[head(u[ids],zq[ids]) for ids in torch.tensor(indices).split(128)]
    return torch.cat([o.slots for o in outputs]),torch.cat([o.embedding for o in outputs])

def create(arch,width,k,decision,seed):
    torch.manual_seed(seed)
    mode,gate,out,joint=ARCHS[arch]
    outdim=k*256 if out=='equal' else (256 if k>1 else 128) if out=='small' else None
    kwargs={'num_slots':k,'output_dim':outdim}
    if mode=='raw': return OptionEmbeddingHead(width,embed_dim=256,mode=mode,**kwargs)
    return OptionEmbeddingHead.from_decision_head(decision,256,mode,gate,**kwargs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    from superembedding.baselines import MODELS
    ap.add_argument('--backend',choices=['omnijev','base','flair',*MODELS],required=True)
    ap.add_argument('--protocol',choices=['v3','v4'],default='v3')
    ap.add_argument('--scales',nargs='+',type=int,default=[25,50,100])
    ap.add_argument('--seeds',nargs='+',type=int,default=[17,42,91])
    ap.add_argument('--lrs',nargs='+',type=float,default=[.0001,.001])
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--architectures',nargs='+',choices=list(ARCHS))
    ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--output-root',default='runs/v3')
    ap.add_argument('--data',help='Override only for isolated recovery verification')
    ap.add_argument('--features',help='Explicit cache directory for isolated recovery verification')
    args=ap.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    # cuBLAS deterministic workspace must be set before first CUDA matmul.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    base=Path(args.output_root)/args.dataset
    meta,records,prov,basis,u,zq,labels,splits,ident=load(args.data or Path('data')/args.dataset,args.features or base/args.backend)
    k=len(basis.directions); device='cuda'; u,zq=u.to(device),zq.to(device)
    val_labels=[labels[i] for i in splits['val']]
    relations=prepare_relations(val_labels,meta['label_keys'],device)
    decision=None
    if args.backend=='omnijev':
        sys.path.insert(0,str(Path('repos/OmniJev').resolve()))
        from mso.head import OptionScorer
        decision=OptionScorer(u.shape[-1]); decision.load_state_dict(torch.load(Path(prov['adapter'])/'head.pt',map_location='cpu',weights_only=True))
    source={'decision_checkpoint':prov.get('adapter','FLAIR' if args.backend=='flair' else None),
            'base_model':prov.get('base',prov.get('model',prov.get('spec',{}).get('repo',prov.get('spec',{}).get('model','FLAIR')))),
            'feature_backend':args.backend,'research_protocol':args.protocol}
    for scale in args.scales:
        train=sorted(splits['train'],key=lambda i:records[i]['train_rank'])[:meta['scales'][str(scale)]]
        pools=ObservedTriplets(labels,train); joint_pools=ObservedTriplets(labels,train,joint=True)
        if args.architectures: archs=args.architectures
        elif args.backend=='base' or args.backend in MODELS: archs=['raw']
        elif args.backend=='flair': archs=['raw'] if scale<100 else ['raw','joint_concat','joint_out_equal','joint_out_small']
        else: archs=list(ARCHS) if scale==100 else ['raw','joint_concat','joint_out_equal','joint_out_small']
        for arch,seed,lr in itertools.product(archs,args.seeds,args.lrs):
            out=base/'suite'/args.backend/f'n{scale}'/f'{arch}_s{seed}_lr{lr:g}'; out.mkdir(parents=True,exist_ok=True)
            config={'protocol':args.protocol,'arch':arch,'backend':args.backend,'scale':scale,'train_count':len(train),
                    'seed':seed,'lr':lr,'steps':args.steps,'batch_size':64,'margin':.2,'data_fingerprint':ident,
                    'basis':asdict(basis),'label_policy':meta.get('label_policy','observed set overlap'),
                    'feature_width':u.shape[-1],'embed_dim':256,'val_every':250,
                    'sampling':'uniform attribute and eligible anchor; positive by observed overlap; observed disjoint negative',
                    'checkpoint_version':1}
            if (out/'validation.json').exists():
                if json.loads((out/'validation.json').read_text())['config']!=json.loads(json.dumps(config)):
                    raise ValueError('Completed run config mismatch')
                print(f'SKIP {out}',flush=True); continue
            random.seed(seed); rng=random.Random(seed)
            head=create(arch,u.shape[-1],k,decision,seed).to(device)
            opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
            joint_loss=ARCHS[arch][-1]
            def validation():
                slots,vectors=project(head,u,zq,splits['val'])
                metrics=evaluate(slots,vectors,meta['label_keys'],relations)
                score=metrics['attributes']['macro']['mAP']
                if joint_loss: score=(score+metrics['joint']['mAP'])/2
                return score,metrics
            state={'best':-1.,'best_step':0,'history':[]}; step0=0
            if (out/'last.pt').exists():
                step0,state=restore_training(out/'last.pt',head,opt,config,rng)
                print(f'RESUME {out} step={step0}',flush=True)
            else:
                score,metrics=validation(); state.update(best=score,best_metrics=metrics)
                head.save(out/'best_head.pt',basis,source)
                save_training(out/'last.pt',head,opt,0,config,rng,state)
            start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            loss_sum=state.get('window_loss_sum',0.); active_sum=state.get('window_active_sum',0.); samples=state.get('window_samples',0)
            for step in range(step0+1,args.steps+1):
                head.train(); ids,attrs=pools.sample(64,rng)
                output=head(u[ids.flatten()],zq[ids.flatten()]); slots=output.slots.reshape(64,3,k,256)
                ai=torch.arange(64,device=device); attrs=attrs.to(device)
                a,p,n=slots[:,0][ai,attrs],slots[:,1][ai,attrs],slots[:,2][ai,attrs]
                terms=F.relu((a*n).sum(-1)-(a*p).sum(-1)+.2); loss=terms.mean()
                if joint_loss:
                    ids,_=joint_pools.sample(64,rng)
                    v=head(u[ids.flatten()],zq[ids.flatten()]).embedding.reshape(64,3,-1)
                    jt=F.relu((v[:,0]*v[:,2]).sum(-1)-(v[:,0]*v[:,1]).sum(-1)+.2)
                    loss=loss+jt.mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
                opt.step(); loss_sum+=loss.item(); active_sum+=(terms>0).float().mean().item(); samples+=1
                if step%250==0 or step==args.steps:
                    score,metrics=validation()
                    if score>state['best']:
                        state.update(best=score,best_step=step,best_metrics=metrics); head.save(out/'best_head.pt',basis,source)
                    item={'step':step,'loss':loss_sum/samples,'active_attribute_triplets':active_sum/samples,'score':score}
                    state['history'].append(item); loss_sum=active_sum=0.; samples=0
                    print(json.dumps({'run':str(out),**item}),flush=True)
                if step%100==0 or step==args.steps or step==args.stop_after:
                    state.update(window_loss_sum=loss_sum,window_active_sum=active_sum,window_samples=samples)
                    save_training(out/'last.pt',head,opt,step,config,rng,state)
                if step==args.stop_after:
                    print('STOPPED; exact resume checkpoint retained.',flush=True); return
            # last.pt retains optimizer/RNG; best_head.pt is independently usable.
            report={'config':config,**state,'test_evaluated':False,
                    'trainable_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
                    'output_dimension':head.output_dim or k*256,
                    'seconds_this_invocation':time.perf_counter()-start,
                    'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20}
            atomic_json(report,out/'validation.json')
            del head,opt

if __name__=='__main__':
    from filelock import FileLock
    lock_parser=argparse.ArgumentParser(add_help=False)
    lock_parser.add_argument('--dataset',required=True)
    lock_parser.add_argument('--backend',required=True)
    lock_parser.add_argument('--output-root',default='runs/v3')
    lock_args,_=lock_parser.parse_known_args()
    lock_path=Path(lock_args.output_root)/lock_args.dataset/f'training_{lock_args.backend}.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock_path)):
        main()
