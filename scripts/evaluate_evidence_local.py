"""CPU pretrained CLIP/SigLIP controls with immutable, resumable vector shards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from PIL import Image
from superembedding.baselines import MODELS
from superembedding.checkpoint import atomic_json,atomic_save,fingerprint


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['clip_b32','siglip2_b16'],required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    from transformers import AutoImageProcessor,AutoTokenizer,CLIPModel,SiglipModel
    from huggingface_hub import snapshot_download
    root=Path('data/evidence_diagnostic_v1');out=Path('runs/v5/evidence')/a.backend
    out.mkdir(parents=True,exist_ok=True)
    raw=(root/'records.jsonl').read_bytes();records=[json.loads(s) for s in raw.decode().splitlines()]
    meta=json.loads((root/'metadata.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=meta['records_sha256']:raise ValueError('Dataset manifest changed')
    images=sorted({r[k] for r in records for k in ['anchor','relevant_change','irrelevant_change']})
    texts=sorted({r['query'] for r in records});spec=MODELS[a.backend]
    folder=snapshot_download(spec['repo'],revision=spec['revision'],
        cache_dir='C:/Users/123/.cache/superembedding/v4/models',local_files_only=True,
        allow_patterns=['config.json','preprocessor_config.json','pytorch_model.bin'] if a.backend=='clip_b32'
        else ['config.json','preprocessor_config.json','model.safetensors'])
    processor=AutoImageProcessor.from_pretrained(folder,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(spec['repo'],revision=spec['revision'],local_files_only=True)
    cls=CLIPModel if a.backend=='clip_b32' else SiglipModel
    model=cls.from_pretrained(folder,local_files_only=True).eval()
    provenance={'spec':spec,'dataset_sha256':meta['records_sha256'],'device':'cpu','dtype':'float32',
                'query':'original v1 query, no extra instruction','candidate':'query-independent image vector',
                'max_text_tokens':77 if a.backend=='clip_b32' else 64,
                'transformers':__import__('transformers').__version__,
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started=time.perf_counter();vectors={}
    for kind,items in [('image',images),('text',texts)]:
        parts=[]
        for start in range(0,len(items),8):
            batch=items[start:start+8];path=out/f'{kind}_{start:04d}.pt'
            sig=fingerprint({'provenance':provenance,'items':batch,
                'image_sha256':[hashlib.sha256((root/p).read_bytes()).hexdigest() for p in batch] if kind=='image' else []})
            if path.exists():
                saved=torch.load(path,weights_only=True)
                if saved['fingerprint']!=sig:raise ValueError('Evidence cache fingerprint mismatch')
                v=saved['vectors']
            else:
                tick=time.perf_counter()
                if kind=='image':
                    decoded=[]
                    for p in batch:
                        with Image.open(root/p) as im:decoded.append(im.convert('RGB'))
                    v=model.get_image_features(**processor(images=decoded,return_tensors='pt'))
                else:
                    inputs=tokenizer(batch,max_length=provenance['max_text_tokens'],truncation=True,
                                     padding='max_length',return_tensors='pt')
                    v=model.get_text_features(**inputs)
                v=v.pooler_output if hasattr(v,'pooler_output') else v
                v=F.normalize(v.float(),dim=-1).cpu()
                if v.ndim!=2 or len(v)!=len(batch) or not torch.isfinite(v).all():raise ValueError('Invalid vectors')
                atomic_save({'vectors':v,'fingerprint':sig,'seconds':time.perf_counter()-tick},path)
            parts.append(v)
            if start%80==0:print(json.dumps({'backend':a.backend,'kind':kind,'encoded':min(start+8,len(items)),'total':len(items)}),flush=True)
        packed=torch.cat(parts);vectors[kind]={item:packed[i] for i,item in enumerate(items)}
    cases=[]
    for r in records:
        av,bv,cv=[vectors['image'][r[k]] for k in ['anchor','relevant_change','irrelevant_change']]
        q=vectors['text'][r['query']];sa,sb,sc=[float(q@v) for v in [av,bv,cv]]
        cases.append({'id':r['id'],'scene_group':r['scene_group'],'attribute':r['attribute'],
            'anchor_margin':sa-sb,'nuisance_margin':sc-sb,'nuisance_query_shift':abs(sa-sc),
            'target_change_distance':float(1-av@bv),'nuisance_distance':float(1-av@cv)})
    metrics={}
    for attr in sorted({r['attribute'] for r in cases}):
        rows=[r for r in cases if r['attribute']==attr];n=len(rows)
        metrics[attr]={'cases':n,'anchor_preference':sum(r['anchor_margin']>0 for r in rows)/n,
            'robust_both_preference':sum(min(r['anchor_margin'],r['nuisance_margin'])>0 for r in rows)/n}
    atomic_json({'backend':a.backend,'provenance':provenance,'dataset_sha256':meta['records_sha256'],
        'metrics':metrics,'cases':cases,'seconds_this_invocation':time.perf_counter()-started,
        'scope':'Frozen no-training diagnostic; CPU timings are not comparable with GPU/API latency'},
        Path('artifacts/results/v5')/f'evidence_{a.backend}.json')
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
