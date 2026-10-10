"""Resumable real-image LoRA + embedding training on 8GB; optional validation selection.

Without --validate-every this remains a training smoke run, not a quality study.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MSO_FLA", "0")
os.environ.setdefault("MSO_BRANCH_CHUNK", "4")

import torch
from superembedding import AttributeBasis, attribute_triplet_loss
from superembedding.omnijev import OmniJevEmbeddingEncoder, load_omnijev
from superembedding.checkpoint import save_training, restore_training, atomic_save, atomic_copy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/fashion600')
    ap.add_argument('--adapter', default='models/OmniJev-0.8B-v1.1')
    ap.add_argument('--base', default='models/Qwen3.5-0.8B')
    ap.add_argument('--head', default='runs/fashion600/gated/best_head.pt')
    ap.add_argument('--output', default='runs/fashion600/lora_smoke')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--accumulation', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--max-pixels', type=int, default=65536)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--checkpoint-every', type=int, default=10)
    ap.add_argument('--stop-after', type=int, default=0)
    ap.add_argument('--validate-every', type=int, default=0, help='0 for smoke; otherwise select by held-out validation macro attribute mAP')
    ap.add_argument('--relocate-source',action='store_true',help='Allow explicit adapter/base path relocation for a published head; resume still checks content hashes')
    args = ap.parse_args()
    if args.steps < 1 or args.accumulation < 1 or args.checkpoint_every < 1 or args.validate_every < 0:
        raise ValueError('steps, accumulation and checkpoint interval must be positive; validation interval nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this local LoRA run')
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    basis = AttributeBasis(tuple(meta['directions']), meta['instruction'])
    records = [json.loads(line) for line in (root/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    val_records = [r for r in records if r['split'] == 'val']
    records = [r for r in records if r['split'] == 'train']
    keys = meta['label_keys']
    pools = []
    for k, key in enumerate(keys):
        for a in records:
            positive = [r for r in records if r['id'] != a['id'] and r['labels'][key] == a['labels'][key]]
            negative = [r for r in records if r['labels'][key] != a['labels'][key]]
            if positive and negative:
                pools.append((k, a, positive, negative))
    if not pools:
        raise ValueError('no eligible training triplets')
    mso = load_omnijev(args.adapter, args.base, max_pixels=args.max_pixels)
    encoder = OmniJevEmbeddingEncoder(mso, basis, tune_lora=True)
    encoder.load_head(args.head,allow_source_relocation=args.relocate_source)
    encoder.train()
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    config={k:v for k,v in vars(args).items() if k not in {'stop_after','output','data','head','adapter','base','relocate_source'}}
    config['data_sha256']=hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest()
    config['head_sha256']=hashlib.sha256(Path(args.head).read_bytes()).hexdigest()
    for label,path in [('adapter',Path(args.adapter)/'adapter_model.safetensors'),('base',Path(args.base)/'source_revision.json')]:
        if path.exists(): config[label+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    name, probe = next((name, p) for name, p in encoder.backbone.named_parameters()
                       if 'lora_B' in name and p.requires_grad)
    before = probe.detach().float().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history = []
    best=-1.; best_step=0; step0=0
    def validate():
        from scripts.train_cached import retrieval_metrics
        encoder.eval()
        parts=[]
        for index,r in enumerate(val_records,1):
            parts.append(encoder.encode({'images':[str(root/r['image'])]}).slots.cpu())
            if index%100==0:
                print(json.dumps({'validation_encoded':index,'total':len(val_records)}),flush=True)
        slots=torch.stack(parts)
        maps={key:{v:i for i,v in enumerate(sorted({r['labels'][key] for r in val_records}))} for key in keys}
        labels=torch.tensor([[maps[key][r['labels'][key]] for key in keys] for r in val_records])
        metrics=retrieval_metrics(slots,labels,keys)
        encoder.train()
        return metrics
    if (output/'last.pt').exists():
        step0,extra=restore_training(output/'last.pt',encoder,optimizer,config,rng)
        history=extra['history']; best=extra['best']; best_step=extra['best_step']
        print(json.dumps({'resume_step':step0}),flush=True)
    else:
        if args.validate_every:
            metrics=validate(); best=metrics['macro']['mAP']
            history.append({'step':0,'val':metrics})
        extra={'history':history,'best':best,'best_step':best_step}
        save_training(output/'last.pt',encoder,optimizer,0,config,rng,extra,trainable_only=True)
        atomic_copy(output/'last.pt',output/'best.pt')
    for step in range(step0+1, args.steps+1):
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(args.accumulation):
            k, a, positives, negatives = rng.choice(pools)
            triplet = (a, rng.choice(positives), rng.choice(negatives))
            embeddings = [encoder({'images': [str(root/r['image'])]}).slots.unsqueeze(0) for r in triplet]
            loss = attribute_triplet_loss(*embeddings, torch.tensor([k], device=mso.dev), margin=args.margin)
            (loss/args.accumulation).backward()
            total += loss.detach().item()/args.accumulation
            del embeddings, loss
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite gradient norm')
        lora_gradient = probe.grad.norm().item() if probe.grad is not None else 0.
        optimizer.step()
        torch.cuda.synchronize()
        entry = {'step': step, 'loss': total, 'gradient_norm': norm.item(),
                 'probe_lora_gradient_norm': lora_gradient,
                 'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                 'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        improved=False
        if args.validate_every and (step%args.validate_every==0 or step==args.steps):
            metrics=validate(); entry['val']=metrics
            if metrics['macro']['mAP']>best:
                best=metrics['macro']['mAP']; best_step=step; improved=True
            print(json.dumps({'step':step,'val':metrics,'best_step':best_step}),flush=True)
        if step%args.checkpoint_every==0 or step==args.steps or step==args.stop_after or improved:
            extra={'history':history,'best':best,'best_step':best_step}
            if improved:
                # Commit best before advancing last's best-step metadata.
                save_training(output/'best.pt',encoder,optimizer,step,config,rng,extra,trainable_only=True)
            save_training(output/'last.pt',encoder,optimizer,step,config,rng,extra,trainable_only=True)
            if step%50==0: atomic_copy(output/'last.pt',output/f'step_{step:06d}.pt')
        if step==args.stop_after:
            print('Paused at completed optimizer step; rerun without --stop-after to resume.',flush=True)
            return
    if args.validate_every:
        saved=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        encoder.load_state_dict(saved['model'],strict=False)
    delta = (probe.detach().float().cpu()-before).abs().max().item()
    reference_image = str(root/records[0]['image'])
    reference = encoder.encode({'images': [reference_image]})
    atomic_save({'image': reference_image, 'slots': reference.slots.cpu(),
                'embedding': reference.embedding.cpu()}, output/'reload_reference.pt')
    adapter_out = output/'adapter'
    encoder.backbone.save_pretrained(adapter_out, safe_serialization=True)
    # Preserve inference-side token embeddings, head and processor; exclude old
    # weights/manifests/calibration claims that do not describe this new run.
    for filename in ['tokenizer.json', 'tokenizer_config.json', 'processor_config.json',
                     'chat_template.jinja', 'new_tok_emb.pt', 'head.pt', 'ord.pt', 'LICENSE']:
        src = Path(args.adapter)/filename
        if src.exists():
            shutil.copy2(src, adapter_out/filename)
    head_meta = json.loads((Path(args.adapter)/'head_meta.json').read_text())
    for key in ['temperatures', 'temperature', 'biases', 'calibration']:
        head_meta.pop(key, None)
    head_meta['embedding_finetuned'] = True
    (adapter_out/'head_meta.json').write_text(json.dumps(head_meta, indent=2), encoding='utf-8')
    encoder.source = {'decision_checkpoint': str(adapter_out), 'base_model': str(Path(args.base).resolve())}
    encoder.save_head(output/'embedding_head.pt')
    report = {'config': vars(args), 'history': history, 'probe_parameter': name,
              'best_validation_mAP':best,'best_step':best_step,'test_evaluated':False,
              'probe_max_abs_weight_change': delta, 'trainable_parameters': sum(p.numel() for p in parameters),
              'seconds': time.perf_counter()-started, 'gpu': torch.cuda.get_device_name(),
              'scope': ('Validation-selected real-image LoRA plus projection; one-seed exploratory follow-up. '
                        if args.validate_every else 'Real-image LoRA plus projection training smoke; retrieval quality not evaluated. ')
                       + 'Inherited decision heads are not recalibrated or validated after LoRA updates.'}
    (output/'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(output), 'lora_weight_change': delta}), flush=True)


if __name__ == '__main__':
    main()
