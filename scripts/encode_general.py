"""Inference entry matching formal general evaluation (image, text, or both)."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--image');ap.add_argument('--text',default='')
    ap.add_argument('--output',required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=ap.parse_args()
    if not a.image and not a.text.strip():raise ValueError('Provide image and/or text')
    if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
    import torch
    from superembedding.general_export import load_bundle_encoder
    from superembedding.checkpoint import atomic_save
    torch.set_num_threads(2);model,bundle=load_bundle_encoder(a.model)
    with torch.no_grad():result=model({'text':a.text,'images':[a.image] if a.image else []})
    atomic_save({'embedding':result.embedding.cpu(),'slots':result.slots.cpu(),'step':bundle['step'],
        'source_checkpoint_sha256':bundle['source_checkpoint_sha256'],'encoder':bundle['encoder']},a.output)
    print(json.dumps({'dimension':result.embedding.numel(),'checkpoint_step':bundle['step'],'output':a.output}))


if __name__=='__main__':main()
