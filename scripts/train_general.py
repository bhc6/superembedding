"""Source-balanced, gradient-cached OmniJev LoRA training with atomic resume."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0')
os.environ.setdefault('MSO_BRANCH_CHUNK','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from superembedding.general import GeneralOmniEncoder, GENERAL_BASIS, cached_backward, positive_mask
from superembedding.omnijev import load_omnijev
from superembedding.checkpoint import atomic_json, atomic_save, save_training, restore_training, atomic_copy


def pair(record, rng):
    text = rng.choice(record['texts'])
    answer = ' '.join(text['assistant'].split())
    if record['task'] == 'caption':
        q = {'text': 'Find an image matching this description: ' + answer}
        d = {'images':[record['image']]}
        key = ('image:'+record['original_rgb_sha256'],'caption:'+answer.casefold())
    else:
        q = {'images':[record['image']], 'text':'Find the answer to this question: '+text['user']}
        d = {'text':answer}
        key = answer.casefold().strip(' .!?')
    return q,d,key


def valid_records(records):
    for record in records:
        texts = [t for t in record['texts'] if t['assistant'].strip() and t['user'].strip()
                 and len(t['assistant']) < 5000]
        if texts:
            yield {**record,'texts':texts}


def make_config(args, raw):
    config = {k:v for k,v in vars(args).items() if k not in {'data','output','stop_after'}}
    config.update(protocol='v5-general-2-full-validation',data_sha256=hashlib.sha256(raw).hexdigest(),
                  basis=list(GENERAL_BASIS.directions),projection='raw-shared-256-no-output-projection',
                  source_balancing='uniform cyclic shuffled source per optimizer step',
                  curriculum='first 256 source row positions for steps 1..500; all rows thereafter',
                  loss='directional multi-positive InfoNCE, tau=.05; same source in-batch negatives',
                  dropout='disabled; replay independently checks exact vectors',
                  transformers=__import__('transformers').__version__,torch=str(torch.__version__))
    for label,path in [('adapter','models/OmniJev-0.8B-v1.1/adapter_model.safetensors'),
                       ('base','models/Qwen3.5-0.8B/source_revision.json')]:
        config[label+'_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config['implementation_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in ['scripts/train_general.py','superembedding/general.py','superembedding/checkpoint.py','superembedding/heads.py']}
    config['deterministic_algorithms']=True
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--seed',type=int,default=20261011)
    ap.add_argument('--max-pixels',type=int,default=65536)
    ap.add_argument('--checkpoint-every',type=int,default=25)
    ap.add_argument('--validate-every',type=int,default=250)
    ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--milestone-every',type=int,default=1000)
    ap.add_argument('--validation-limit',type=int,default=0,help='0 uses every validation image per source')
    args = ap.parse_args()
    if min(args.steps,args.batch_size,args.checkpoint_every,args.validate_every,args.milestone_every) < 1:
        raise ValueError('Training and checkpoint intervals must be positive')
    root=Path(args.data); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    raw = (root/'records.jsonl').read_bytes()
    records = list(valid_records([json.loads(line) for line in raw.decode('utf-8').splitlines()]))
    sources = sorted({r['source'] for r in records if r['split']=='train'})
    pools = {s:[r for r in records if r['split']=='train' and r['source']==s] for s in sources}
    # Only validation selects checkpoints; OOD remains unopened by this script.
    if args.validation_limit<0:raise ValueError('validation-limit must be nonnegative')
    val = {s:sorted([r for r in records if r['split']=='val' and r['source']==s],key=lambda r:r['id'])[:args.validation_limit or None] for s in sources}
    config=make_config(args,raw)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:
        raise ValueError('Existing run configuration differs; explicit migration/new directory required')
    atomic_json(config,output/'config.json')
    rng=random.Random(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=args.max_pixels)
    encoder=GeneralOmniEncoder(mso,GENERAL_BASIS,embed_dim=256,mode='raw',tune_lora=True)
    # eval disables LoRA dropout, but all LoRA gradients remain enabled.
    encoder.eval()
    params=[p for p in encoder.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    encode=lambda obj:encoder(obj).embedding
    step0=0; extra={'best_validation':-1.,'best_step':0,'source_queue':[]}
    if (output/'last.pt').exists():
        step0,extra=restore_training(output/'last.pt',encoder,optimizer,config,rng)
        print(json.dumps({'resumed_step':step0}),flush=True)
    def checkpoint(step, milestone=False):
        # Last + temporary + optional immutable milestone fit before writing.
        if shutil.disk_usage(output).free < 550*2**20:
            raise RuntimeError('Less than 550 MiB free for safe atomic checkpoint')
        save_training(output/'last.pt',encoder,optimizer,step,config,rng,extra,trainable_only=True)
        if milestone:
            # Preserve every declared milestone's trained weights; full optimizer
            # and RNG remain in last.pt and validation-best.pt to bound disk use.
            names={n for n,p in encoder.named_parameters() if p.requires_grad}
            atomic_save({'model':{n:p.detach().cpu().clone() for n,p in encoder.state_dict().items() if n in names},
                         'step':step,'config':config,'scope':'inference milestone; resume uses last.pt or best.pt'},
                        output/f'step_{step:06d}_weights.pt')
    def validate():
        scores={}
        with torch.no_grad():
            for source,rows in val.items():
                if len(rows)<2: continue
                pairs=[pair(r,random.Random(r['id'])) for r in rows]
                q=torch.stack([encode(p[0]) for p in pairs]); d=torch.stack([encode(p[1]) for p in pairs])
                ranks=(q@d.T).argmax(-1).tolist()
                mask=positive_mask([p[2] for p in pairs],'cpu')
                scores[source]=sum(bool(mask[i,j]) for i,j in enumerate(ranks))/len(rows)
                print(json.dumps({'validation_source':source,'queries':len(rows),'r1':scores[source]}),flush=True)
        return {'macro_r1':sum(scores.values())/len(scores),'per_source_r1':scores,
                'per_source_queries':{s:len(r) for s,r in val.items()},
                'scope':'All held validation images per source' if not args.validation_limit else f'First {args.validation_limit} validation images per source'}
    if step0 == 0 and not (output/'last.pt').exists():
        checkpoint(0)
    probe=next(p for n,p in encoder.backbone.named_parameters() if 'lora_B' in n and p.requires_grad)
    first_weight=probe.detach().float().clone()
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    last_saved=step0
    try:
        for step in range(step0+1,args.steps+1):
            tick=time.perf_counter()
            if not extra['source_queue']:
                extra['source_queue']=sources.copy(); rng.shuffle(extra['source_queue'])
            source=extra['source_queue'].pop()
            pool=[r for r in pools[source] if step>500 or r['order']<256]
            if len(pool)<args.batch_size: raise ValueError('Insufficient distinct source images')
            for _ in range(32):
                batch=rng.sample(pool,args.batch_size)
                pairs=[pair(r,rng) for r in batch]
                if not positive_mask([p[2] for p in pairs],'cpu').all():break
            else:raise ValueError('No negative target after 32 deterministic batch resamples')
            optimizer.zero_grad(set_to_none=True)
            loss,replay_error=cached_backward(encode,[p[0] for p in pairs],[p[1] for p in pairs],[p[2] for p in pairs])
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
            warmup=min(100,max(1,args.steps//10))
            scale=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,args.steps-warmup)))
            for group in optimizer.param_groups: group['lr']=args.lr*scale
            lora_grad=probe.grad.float().norm().item() if probe.grad is not None else 0.
            optimizer.step(); torch.cuda.synchronize()
            entry={'step':step,'source':source,'loss':loss,'lr':args.lr*scale,'gradient_norm':norm.item(),
                   'lora_probe_grad_norm':lora_grad,'lora_probe_delta_from_launch':(probe.detach().float()-first_weight).abs().max().item(),
                   'gradient_replay_max_error':replay_error,'seconds':time.perf_counter()-tick,
                   'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20}
            if step%args.validate_every==0 or step==args.steps:
                entry['validation']=validate()
                if entry['validation']['macro_r1']>extra['best_validation']:
                    extra['best_validation']=entry['validation']['macro_r1']; extra['best_step']=step
                    save_training(output/'best.pt',encoder,optimizer,step,config,rng,extra,trainable_only=True)
            with (output/'history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(entry)+'\n'); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(entry),flush=True)
            if step%args.checkpoint_every==0 or step in {args.stop_after,args.steps}:
                checkpoint(step,step%args.milestone_every==0); last_saved=step
            atomic_json({**entry,'pid':os.getpid(),'state':'running','last_checkpoint_step':last_saved,
                         'completed_this_launch':step-step0,'elapsed_seconds':time.perf_counter()-started},output/'status.json')
            if step==args.stop_after or (output/'STOP_AFTER_CHECKPOINT').exists():
                if last_saved!=step: checkpoint(step)
                atomic_json({'state':'paused','step':step,'last_checkpoint_step':step},output/'status.json')
                return
        encoder.save_head(output/'embedding_head.pt')
        atomic_json({'state':'completed','step':args.steps,'best_step':extra['best_step'],
                     'quality_claim':'No publication or SOTA claim; external evaluation required.'},output/'status.json')
    except BaseException as exc:
        atomic_json({'state':'failed','last_checkpoint_step':last_saved,'error_type':type(exc).__name__,
                     'message':str(exc),'pid':os.getpid()},output/'status.json')
        raise


if __name__=='__main__':
    from filelock import FileLock
    lock_parser=argparse.ArgumentParser(add_help=False);lock_parser.add_argument('--output',required=True)
    lock_args,_=lock_parser.parse_known_args();Path(lock_args.output).mkdir(parents=True,exist_ok=True)
    with FileLock(str(Path(lock_args.output)/'training.lock'),timeout=0):main()
