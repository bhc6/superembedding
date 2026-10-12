"""Replay a real optimizer update twice from the same saved model/optimizer/RNG."""
import os
os.environ['MSO_FLA']='0';os.environ['MSO_BRANCH_CHUNK']='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_general import pair
from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS,cached_backward
from superembedding.omnijev import load_omnijev
from superembedding.checkpoint import restore_training,atomic_json

def digest(model,optimizer,rng):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
    for state in optimizer.state.values():
        for key,value in sorted(state.items()):
            h.update(key.encode())
            if isinstance(value,torch.Tensor):h.update(value.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:h.update(str(value).encode())
    h.update(repr(rng.getstate()).encode());h.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():h.update(state.numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data',required=True)
    args=ap.parse_args();run=Path(args.run)
    config=json.loads((run/'config.json').read_text())
    records=[json.loads(s) for s in (Path(args.data)/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=config['max_pixels'])
    model=GeneralOmniEncoder(mso,GENERAL_BASIS,mode='raw',tune_lora=True);model.eval()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=.01);rng=random.Random()
    results=[];first=None;differences={}
    for trial in range(2):
        step,extra=restore_training(run/'last.pt',model,optimizer,config,rng)
        source=extra['source_queue'].pop()
        pool=[r for r in records if r['split']=='train' and r['source']==source]
        batch=rng.sample(pool,config['batch_size']);pairs=[pair(r,rng) for r in batch]
        optimizer.zero_grad(set_to_none=True)
        loss,error=cached_backward(lambda obj:model(obj).embedding,[p[0] for p in pairs],[p[1] for p in pairs],[p[2] for p in pairs])
        torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        results.append({'trial':trial,'start_step':step,'loss':loss,'replay_error':error,
                        'state_sha256':digest(model,optimizer,rng)})
        current={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        if first is None:first=current
        else:
            differences={n:(current[n].float()-v.float()).abs().max().item() for n,v in first.items() if not torch.equal(current[n],v)}
    report={'passed':results[0]['state_sha256']==results[1]['state_sha256'],'trials':results,'parameter_differences':differences,
            'scope':'Real image/text LoRA, optimizer and RNG update replay from saved checkpoint; no quality claim'}
    atomic_json(report,'artifacts/results/v5/general_resume_audit.json')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('Real resume replay differs')

if __name__=='__main__':main()
