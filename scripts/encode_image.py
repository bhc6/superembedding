"""Use a saved embedding head to encode a new image into ordered slots."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MSO_FLA", "0")
os.environ.setdefault("MSO_BRANCH_CHUNK", "4")

import torch
from superembedding import OptionEmbeddingHead
from superembedding.omnijev import OmniJevEmbeddingEncoder, load_omnijev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--head", default="runs/fashion600/gated/best_head.pt")
    ap.add_argument("--output", default="runs/encoded_image.pt")
    ap.add_argument("--max-pixels", type=int, default=65536)
    ap.add_argument('--adapter', help='Explicit local location of the same adapter, for portable published checkpoints')
    ap.add_argument('--base', help='Explicit local location of the pinned base model')
    args = ap.parse_args()
    head, basis, source = OptionEmbeddingHead.load(args.head)
    from superembedding.general import GENERAL_BASIS
    if basis==GENERAL_BASIS:
        raise ValueError('V5 general head alone omits trained LoRA and requires GeneralOmniEncoder. Use encode_general.py --model export/general_model.pt.')
    if source.get('feature_backend') in {'omnijev_lora_disabled','base','flair'} or source.get('decision_checkpoint')=='FLAIR':
        raise ValueError('This baseline head requires its matching feature extractor, not the OmniJev adapter encoder')
    if not source.get('decision_checkpoint') or not source.get('base_model'):
        raise ValueError('checkpoint must record its OmniJev adapter and base model')
    mso = load_omnijev(args.adapter or source['decision_checkpoint'], args.base or source['base_model'], max_pixels=args.max_pixels)
    encoder = OmniJevEmbeddingEncoder(mso, basis, embed_dim=head.embed_dim,
                                      mode=head.mode, train_gate=head.train_gate)
    if args.adapter or args.base:
        # Explicit path relocation; users must retain the model revisions in the model card.
        encoder.embedding_head = head.to(next(encoder.backbone.parameters()).device)
    else:
        encoder.load_head(args.head)
    result = encoder.encode({'images': [str(Path(args.image).resolve())]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'slots': result.slots.cpu(), 'embedding': result.embedding.cpu(),
                'directions': list(basis.directions), 'instruction': basis.instruction,
                'head': str(Path(args.head).resolve()), 'image': str(Path(args.image).resolve())}, output)
    print(json.dumps({'slots': list(result.slots.shape), 'embedding': list(result.embedding.shape),
                      'norm': result.embedding.norm().item(), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
