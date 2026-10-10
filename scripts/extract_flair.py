"""Official FLAIR weights, fixed direction queries, reusable image slot vectors."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos/FLAIR/src'))
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import flair
from superembedding.checkpoint import atomic_save


def main():
    torch.set_num_threads(4)
    root=Path('data/fashion2400'); out=Path('runs/fashion2400/flair'); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'metadata.json').read_text())
    revision='0a306fd7060ba3076da3c9ee75b569e6aabea6c6'
    path=hf_hub_download('xiaorui638/flair','flair-merged30m.pt',revision=revision,local_dir='models/FLAIR')
    provenance={'model':'xiaorui638/flair','revision':revision,'weights':'flair-merged30m.pt',
                'basis':{'directions':meta['directions'],'instruction':meta['instruction']},
                'preprocessing':'official ViT-B-16-FLAIR eval transform (224px)',
                'prompts':meta['directions'],'dataset_revision':meta['revision']}
    manifest=out/'provenance.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance: raise ValueError('cache mismatch')
    manifest.write_text(json.dumps(provenance,indent=2))
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    todo=[r for r in records if not (out/f"{r['id']}.pt").exists()]
    if not todo:
        print('All FLAIR features already cached; retaining original extraction metrics.')
        return
    # Official checkpoint includes NumPy scalar metadata. Keep weights_only=True.
    with torch.serialization.safe_globals([(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]):
        model,_,preprocess=flair.create_model_and_transforms('ViT-B-16-FLAIR',pretrained=path)
    model=model.cuda().eval()
    tokenizer=flair.get_tokenizer('ViT-B-16-FLAIR')
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        text,_=model.encode_text(tokenizer(meta['directions']).cuda())
        text=model.text_post(text)
        for offset in range(0,len(todo),16):
            rows=todo[offset:offset+16]
            images=torch.stack([preprocess(Image.open(root/r['image']).convert('RGB')) for r in rows]).cuda()
            global_tokens,local_tokens=model.encode_image(images)
            local_tokens=model.image_post(local_tokens)
            global_tokens=model.image_post(global_tokens)
            slots=model.visual_proj(text.unsqueeze(0).expand(len(rows),-1,-1),local_tokens,local_tokens)
            for r,u,zq in zip(rows,slots,global_tokens):
                atomic_save({'u':u.float().cpu(),'zq':zq.float().cpu(),'id':r['id'],
                             'image_sha256':r['image_sha256']},out/f"{r['id']}.pt")
            if offset%160==0: print(json.dumps({'done':min(offset+16,len(todo)),'total':len(todo)}),flush=True)
    (out/'extraction.json').write_text(json.dumps({'seconds':time.perf_counter()-start,
        'count':len(todo),'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20},indent=2))


if __name__=='__main__': main()
