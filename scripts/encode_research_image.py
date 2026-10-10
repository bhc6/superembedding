"""Encode one image with any released v3 backend, preserving its fixed basis."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from superembedding import OptionEmbeddingHead
from superembedding.checkpoint import atomic_save

@torch.no_grad()
def encode(args):
    torch.set_num_threads(4)
    head,basis,source=OptionEmbeddingHead.load(args.head);head=head.cuda().eval()
    backend=source.get('feature_backend')
    if source.get('research_protocol')!='v3' or backend not in ('omnijev','base','flair'):
        raise ValueError('Use a v3 checkpoint; legacy OmniJev heads use encode_image.py')
    if backend in ('omnijev','base'):
        from superembedding.omnijev import load_omnijev,OmniJevEmbeddingEncoder
        from superembedding.batched import prepare_image,encode_prepared
        mso=load_omnijev(args.adapter,args.base,max_pixels=65536)
        encoder=OmniJevEmbeddingEncoder(mso,basis).eval()
        item=prepare_image(encoder,Path(args.image).resolve())
        with encoder.backbone.disable_adapter() if backend=='base' else nullcontext():
            # Match the batch geometry used for training features. Repeated
            # identical images keep state isolation and avoid amplified BF16
            # batch-1 rounding differences, especially in the base ablation.
            u,zq=encode_prepared(encoder,[item]*4);u,zq=u[0],zq[0]
    else:
        import numpy as np
        from PIL import Image
        from huggingface_hub import hf_hub_download
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos/FLAIR/src'))
        import flair
        path=hf_hub_download('xiaorui638/flair','flair-merged30m.pt',revision='0a306fd7060ba3076da3c9ee75b569e6aabea6c6',local_dir=args.flair)
        with torch.serialization.safe_globals([(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]):
            model,_,preprocess=flair.create_model_and_transforms('ViT-B-16-FLAIR',pretrained=path)
        model=model.cuda().eval();tokenizer=flair.get_tokenizer('ViT-B-16-FLAIR')
        text,_=model.encode_text(tokenizer(list(basis.directions)).cuda());text=model.text_post(text)
        with Image.open(args.image) as image: pixels=preprocess(image.convert('RGB')).unsqueeze(0).cuda()
        global_tokens,local_tokens=model.encode_image(pixels)
        local_tokens=model.image_post(local_tokens);global_tokens=model.image_post(global_tokens)
        u=model.visual_proj(text.unsqueeze(0),local_tokens,local_tokens)[0].float();zq=global_tokens[0].float()
    result=head(u,zq)
    payload={'slots':result.slots.cpu(),'embedding':result.embedding.cpu(),
             'directions':list(basis.directions),'instruction':basis.instruction,'backend':backend,
             'head':str(Path(args.head).resolve()),'image':str(Path(args.image).resolve())}
    atomic_save(payload,args.output)
    print(json.dumps({'backend':backend,'slots':list(result.slots.shape),'embedding':list(result.embedding.shape),
                      'output':str(Path(args.output).resolve())},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--head',required=True);ap.add_argument('--image',required=True)
    ap.add_argument('--output',default='runs/v3_encoded_image.pt')
    ap.add_argument('--adapter',default='models/OmniJev-0.8B-v1.1');ap.add_argument('--base',default='models/Qwen3.5-0.8B')
    ap.add_argument('--flair',default='models/FLAIR');encode(ap.parse_args())
