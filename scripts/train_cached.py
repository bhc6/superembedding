"""Legacy fashion600 development trainer (no resume); use experiment_suite.py for new runs.

retrieval_metrics remains the shared evaluation implementation.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from superembedding import AttributeBasis, OptionEmbeddingHead, attribute_triplet_loss


@torch.no_grad()
def retrieval_metrics(slots, labels, label_keys):
    """Within-split image retrieval; exclude self; skip queries with no positives.

    Labels, not the embedding instruction, determine positives for each slot.
    Return per-attribute coverage so singleton exclusion stays visible.
    """
    output = {}
    n = len(labels)
    for k, key in enumerate(label_keys):
        similarities = slots[:, k].float() @ slots[:, k].float().T
        similarities.fill_diagonal_(-torch.inf)
        values = labels[:, k].to(slots.device)
        positive = values[:, None] == values[None, :]
        positive.fill_diagonal_(False)
        eligible = positive.any(dim=1)
        if not eligible.any():
            output[key] = {"eligible_queries": 0, "total_queries": n, "R@1": None, "R@5": None, "mAP": None}
            continue
        order = similarities.argsort(dim=1, descending=True)[:, :n-1]
        hits = positive.gather(1, order)
        ranks = torch.arange(1, n, device=slots.device)
        precision = hits.cumsum(dim=1) / ranks
        average_precision = (precision * hits).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        output[key] = {
            "eligible_queries": int(eligible.sum()), "total_queries": n,
            "R@1": hits[eligible, 0].float().mean().item(),
            "R@5": hits[eligible, :min(5, n-1)].any(dim=1).float().mean().item(),
            "mAP": average_precision[eligible].mean().item(),
        }
    valid = [v for v in output.values() if v["mAP"] is not None]
    output["macro"] = {metric: sum(v[metric] for v in valid)/len(valid) if valid else None
                       for metric in ["R@1", "R@5", "mAP"]}
    return output


@torch.no_grad()
def project(head, u, zq, indices, device):
    head.eval()
    return torch.cat([
        head(u[part].to(device), zq[part].to(device)).slots.cpu()
        for part in indices.split(64)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fashion600")
    ap.add_argument("--features", default="runs/fashion600/features")
    ap.add_argument("--adapter", default="models/OmniJev-0.8B-v1.1")
    ap.add_argument("--output", default="runs/fashion600/gated")
    ap.add_argument("--mode", choices=["gated", "raw"], default="gated")
    ap.add_argument("--train-gate", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    root, cache, out = Path(args.data), Path(args.features), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    provenance = json.loads((cache/"provenance.json").read_text(encoding="utf-8"))
    basis = AttributeBasis(tuple(metadata["directions"]), metadata["instruction"])
    if AttributeBasis(**provenance["basis"]) != basis:
        raise ValueError("feature cache basis does not match data")
    if Path(provenance["adapter"]).resolve() != Path(args.adapter).resolve():
        raise ValueError("cached features and decision gate must use the same adapter")
    records = [json.loads(line) for line in (root/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    features = []
    for r in records:
        payload = torch.load(cache/f"{r['id']}.pt", weights_only=True)
        if payload["id"] != r["id"] or payload["image_sha256"] != r["image_sha256"]:
            raise ValueError("cached features do not match the dataset image")
        features.append(payload)
    u = torch.stack([r["u"] for r in features])
    zq = torch.stack([r["zq"] for r in features])
    if u.shape[1] != len(basis.directions):
        raise ValueError("cached slot count does not match the basis")
    label_keys = metadata["label_keys"]
    label_maps = {key: {v: i for i, v in enumerate(sorted({r['labels'][key] for r in records}))}
                  for key in label_keys}
    labels = torch.tensor([[label_maps[key][r['labels'][key]] for key in label_keys] for r in records])
    split = {name: torch.tensor([i for i, r in enumerate(records) if r['split'] == name])
             for name in ['train', 'val', 'test']}
    if any(len(indices) < 2 for indices in split.values()):
        raise ValueError("train, val and test require at least two images")
    # Both the state features and gate weights come from the same OmniJev adapter.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"repos"/"OmniJev"))
    from mso.head import OptionScorer
    decision = OptionScorer(u.shape[-1])
    state = torch.load(Path(args.adapter)/"head.pt", map_location="cpu", weights_only=True)
    decision.load_state_dict(state, strict=True)
    head = OptionEmbeddingHead.from_decision_head(
        decision, args.embed_dim, args.mode, args.train_gate,
    ).to(device)
    source = {"decision_checkpoint": str(Path(args.adapter).resolve()), "base_model": provenance["base"]}
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    # All triplet pools are built strictly from the training split.
    pools = []
    train_ids = split['train'].tolist()
    for k in range(len(label_keys)):
        by_value = {}
        for i in train_ids:
            by_value.setdefault(int(labels[i, k]), []).append(i)
        pool = []
        for value, group in by_value.items():
            negatives = [i for i in train_ids if int(labels[i, k]) != value]
            if len(group) >= 2 and negatives:
                pool.extend((i, [p for p in group if p != i], negatives) for i in group)
        if not pool:
            raise ValueError(f"no training triplets for {label_keys[k]}")
        pools.append(pool)
    def evaluate(indices):
        slots = project(head, u, zq, indices, device)
        return retrieval_metrics(slots, labels[indices], label_keys)
    initial_val = evaluate(split['val'])
    best = initial_val['macro']['mAP']
    if best is None:
        raise ValueError("validation has no eligible retrieval queries")
    best_step = 0
    head.save(out/'best_head.pt', basis, source)
    history = []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps+1):
        head.train()
        triples, attrs = [], []
        for _ in range(args.batch_size):
            k = rng.randrange(len(pools))
            a, positives, negatives = rng.choice(pools[k])
            triples.append((a, rng.choice(positives), rng.choice(negatives)))
            attrs.append(k)
        ids = torch.tensor(triples)
        flat = ids.flatten()
        slots = head(u[flat].to(device), zq[flat].to(device)).slots
        slots = slots.reshape(args.batch_size, 3, len(label_keys), args.embed_dim)
        loss = attribute_triplet_loss(slots[:, 0], slots[:, 1], slots[:, 2],
                                      torch.tensor(attrs, device=device, dtype=torch.long))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            metrics = evaluate(split['val'])
            score = metrics['macro']['mAP']
            if score > best:
                best, best_step = score, step
                head.save(out/'best_head.pt', basis, source)
            entry = {'step': step, 'loss': loss.item(), 'val': metrics, 'best_step': best_step}
            history.append(entry)
            print(json.dumps({'step': step, 'loss': loss.item(), 'val_macro': metrics['macro'],
                              'best_step': best_step}), flush=True)
    head, _, _ = OptionEmbeddingHead.load(out/'best_head.pt', expected_basis=basis)
    head = head.to(device)
    # Test is evaluated once, after selecting solely on validation mAP.
    final_test = evaluate(split['test'])
    raw_slots = F.normalize(u[split['test']].float(), dim=-1)
    raw_test = retrieval_metrics(raw_slots, labels[split['test']], label_keys)
    report = {
        'config': vars(args), 'device': str(device), 'basis': provenance['basis'],
        'trainable_parameters': sum(p.numel() for p in head.parameters() if p.requires_grad),
        'splits': {k: len(v) for k, v in split.items()}, 'selection': 'highest validation macro mAP',
        'initial_val': initial_val, 'best_step': best_step, 'best_val_macro_mAP': best,
        'test': final_test, 'raw_backbone_test': raw_test, 'history': history,
        'training_and_evaluation_seconds': time.perf_counter()-started,
        'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
        'limitations': 'One seed; development subset; seen attribute values and combinations; '
                       'near duplicates not excluded. Raw hidden baseline has a different vector width. '
                       'No backbone LoRA updates; projection and optional copied gate only.',
    }
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'test': final_test, 'raw_backbone_test': raw_test}, indent=2))


if __name__ == '__main__':
    main()
