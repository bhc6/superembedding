"""Encode a new image with a released v4 head and its pinned backend/prompt basis."""
import argparse
import hashlib
import io
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
import torch
from PIL import Image
from superembedding import OptionEmbeddingHead
from superembedding.baselines import MODELS,instruction
from superembedding.checkpoint import atomic_save
from scripts.extract_baselines import api_key,api_batch,local_model

@torch.no_grad()
def encode_features(images,backend,basis,cache,loaded=None):
    spec=MODELS[backend]
    if 'model' in spec:
        records=[]
        for i,path in enumerate(images):
            with Image.open(path) as im:
                im=im.convert('RGB');digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            records.append({'id':str(i),'image':str(Path(path).resolve()),'image_sha256':digest})
        with httpx.Client(headers={'Authorization':'Bearer '+api_key()},timeout=120) as client:
            if spec.get('conditioned'):
                vectors=torch.stack([api_batch(client,spec,Path('.'),records,instruction(d))[0] for d in basis.directions],1)
                return vectors,vectors.mean(1)
            vector=api_batch(client,spec,Path('.'),records,None)[0]
    else:
        model,processor=loaded or local_model(backend,cache)
        ims=[Image.open(p).convert('RGB') for p in images]
        # Use the training batch size. Independent repeated rows do not exchange attention.
        count=len(ims)
        if count>32:raise ValueError('Encode at most 32 images per call')
        ims+= [ims[-1]]*(32-count)
        inp=processor(images=ims,return_tensors='pt').to('cuda')
        if backend=='dinov2_small':vector=model(**inp).last_hidden_state[:,0]
        else:
            result=model.get_image_features(**inp)
            vector=result.pooler_output if hasattr(result,'pooler_output') else result
        vector=vector[:count].float().cpu()
    return vector[:,None].expand(-1,len(basis.directions),-1),vector

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--head',required=True);ap.add_argument('--image',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--model-cache',default='models/v4');a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    head,basis,source=OptionEmbeddingHead.load(a.head)
    backend=source['feature_backend']
    u,zq=encode_features([a.image],backend,basis,a.model_cache)
    with torch.no_grad():output=head.eval()(u,zq)
    atomic_save({'slots':output.slots[0],'embedding':output.embedding[0],
        'backend':backend,'basis':{'directions':basis.directions,'instruction':basis.instruction}},a.output)
    print(f'Saved {len(basis.directions)} slots and {output.embedding.shape[-1]}d global embedding')

if __name__=='__main__':main()
