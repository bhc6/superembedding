"""Complete Urban1k retrieval with official FLAIR weights and scoring functions."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos/FLAIR/src'))
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from superembedding.checkpoint import atomic_json,atomic_save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['flair','clip_b32','siglip2_b16'],required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    root=Path('C:/Users/123/.cache/superembedding/v5/urban1k')
    rows=json.loads((root/'annotations/annotations.json').read_text(encoding='utf-8'))['annotations']
    if len(rows)!=1000:raise ValueError('Full Urban1k required')
    out=Path('runs/v5/urban1k')/args.backend;out.mkdir(parents=True,exist_ok=True)
    from flair.train import compute_retrieval
    started=time.perf_counter()
    if args.backend=='flair':
        import flair
        with torch.serialization.safe_globals([(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]):
            model,_,preprocess=flair.create_model_and_transforms('ViT-B-16-FLAIR',pretrained='models/FLAIR/flair-merged30m.pt')
        model=model.cuda().eval();tokenizer=flair.get_tokenizer('ViT-B-16-FLAIR')
        from flair.train import compute_similarity_scores_attn_pool
        with torch.no_grad():
            texts=[]
            for start in range(0,1000,32):
                tokens=tokenizer([r['caption'] for r in rows[start:start+32]]).cuda()
                t,_=model.encode_text(tokens);texts.append(model.text_post(t).cpu())
            t=torch.cat(texts);features=F.normalize(t,dim=-1)
            local_scores=[];global_scores=[];consistency=None
            for i,row in enumerate(rows):
                path=out/f'image_{i:04d}.pt'
                if path.exists():saved=torch.load(path,weights_only=True)
                else:
                    im=preprocess(Image.open(root/'images'/row['image']).convert('RGB')).unsqueeze(0).cuda()
                    g,local=model.encode_image(im);g=model.image_post(g);local=model.image_post(local)
                    conditioned=model.visual_proj(t.cuda().unsqueeze(0),local,local)
                    native=(F.normalize(conditioned,dim=-1)[0]*features.cuda()).sum(-1).cpu()*model.logit_scale.exp().cpu()
                    global_sim=(F.normalize(g,dim=-1).reshape(1,-1).cpu()@features.T)[0]*model.logit_scale.exp().cpu()
                    saved={'native':native,'global':global_sim};atomic_save(saved,path)
                    if i==0:
                        expected,_=compute_similarity_scores_attn_pool(model,[(im,torch.tensor([0]))],features,
                            t.cuda(),torch.device('cuda'),torch.float32,nullcontext,'inference_with_flair')
                        consistency=(expected[0]-native).abs().max().item()
                        if consistency>1e-4:raise RuntimeError('FLAIR official scoring mismatch')
                        atomic_json({'max_abs_score_error':consistency,'function':'flair.train.compute_similarity_scores_attn_pool'},out/'official_equivalence.json')
                local_scores.append(saved['native']);global_scores.append(saved['global'])
                if (i+1)%100==0:print(json.dumps({'backend':'flair','images':i+1}),flush=True)
            scores={'native_text_conditioned':torch.stack(local_scores),'global':torch.stack(global_scores)}
        provenance={'repo_revision':'0f9ce05886d1ee0aa2c628653104cce77a1fff14','model':'xiaorui638/flair',
                    'model_revision':'0a306fd7060ba3076da3c9ee75b569e6aabea6c6','weight':'flair-merged30m.pt',
                    'preprocessing':'official ViT-B-16-FLAIR tokenizer and image transform',
                    'scoring':'all 1000 captions attention-pool each image; exact official scorer equivalence audited',
                    'precision':'float32 (official script defaults AMP); no retraining',
                    'memory_caveat':'Native FLAIR conditioning depends on every query, not a fixed cached image vector'}
    else:
        from scripts.extract_baselines import local_model
        from superembedding.baselines import MODELS
        from transformers import AutoTokenizer
        spec=MODELS[args.backend]
        model,processor=local_model(args.backend,'C:/Users/123/.cache/superembedding/v4/models')
        tokenizer=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['revision'])
        image_parts=[];text_parts=[]
        with torch.no_grad():
            for start in range(0,1000,16):
                batch=rows[start:start+16]
                inp=tokenizer([r['caption'] for r in batch],max_length=77 if args.backend=='clip_b32' else 64,
                              truncation=True,padding='max_length',return_tensors='pt').to('cuda')
                t=model.get_text_features(**inp);t=t.pooler_output if hasattr(t,'pooler_output') else t
                images=[Image.open(root/'images'/r['image']).convert('RGB') for r in batch]
                im=model.get_image_features(**processor(images=images,return_tensors='pt').to('cuda'))
                im=im.pooler_output if hasattr(im,'pooler_output') else im
                image_parts.append(F.normalize(im.float(),dim=-1).cpu());text_parts.append(F.normalize(t.float(),dim=-1).cpu())
        scores={'native_global':torch.cat(image_parts)@torch.cat(text_parts).T};provenance=spec
    metrics={name:compute_retrieval(s,{i:i for i in range(1000)},{i:[i] for i in range(1000)}) for name,s in scores.items()}
    metrics={name:{key:float(value) for key,value in report.items()} for name,report in metrics.items()}
    result={'backend':args.backend,'dataset':json.loads((root/'source.json').read_text()),'queries':1000,'candidates':1000,
            'metrics':metrics,'provenance':provenance,'seconds_this_invocation':time.perf_counter()-started,
            'annotation_sha256':hashlib.sha256((root/'annotations/annotations.json').read_bytes()).hexdigest(),
            'scope':'Full official FLAIR Urban1k annotations; pretrained-weight inference reproduction, not full training reproduction.'}
    atomic_save(scores,out/'scores.pt');atomic_json(result,Path('artifacts/results/v5')/f'urban1k_{args.backend}.json')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
