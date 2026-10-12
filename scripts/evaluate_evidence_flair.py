"""Official FLAIR conditional/global controls on evidence interventions, CPU only."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos/FLAIR/src'))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from superembedding.checkpoint import atomic_save,atomic_json,fingerprint


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    import flair
    from flair.train import compute_similarity_scores_attn_pool
    root=Path('data/evidence_diagnostic_v1');out=Path('runs/v5/evidence/flair');out.mkdir(parents=True,exist_ok=True)
    raw=(root/'records.jsonl').read_bytes();records=[json.loads(s) for s in raw.decode().splitlines()]
    meta=json.loads((root/'metadata.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=meta['records_sha256']:raise ValueError('Dataset manifest changed')
    images=sorted({r[k] for r in records for k in ['anchor','relevant_change','irrelevant_change']})
    texts=sorted({r['query'] for r in records});text_index={s:i for i,s in enumerate(texts)}
    weights=Path('models/FLAIR/flair-merged30m.pt')
    provenance={'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'model_revision':'0a306fd7060ba3076da3c9ee75b569e6aabea6c6',
        'repo_revision':'0f9ce05886d1ee0aa2c628653104cce77a1fff14',
        'dataset_sha256':meta['records_sha256'],'device':'cpu','dtype':'float32',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'query':'original v1 text; official tokenizer; no extra instruction'}
    with torch.serialization.safe_globals([(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),
            np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]):
        model,_,preprocess=flair.create_model_and_transforms('ViT-B-16-FLAIR',pretrained=str(weights))
    model=model.cpu().eval();tokenizer=flair.get_tokenizer('ViT-B-16-FLAIR')
    parts=[]
    for start in range(0,len(texts),8):
        t,_=model.encode_text(tokenizer(texts[start:start+8]));parts.append(model.text_post(t))
    t=torch.cat(parts);features=F.normalize(t,dim=-1);scores={};started=time.perf_counter()
    for i,name in enumerate(images):
        path=out/f'image_{i:04d}.pt'
        sig=fingerprint({'provenance':provenance,'image':name,
            'image_sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),'texts':texts})
        if path.exists():
            saved=torch.load(path,weights_only=True)
            if saved['fingerprint']!=sig:raise ValueError('FLAIR cache mismatch')
        else:
            with Image.open(root/name) as image:im=preprocess(image.convert('RGB')).unsqueeze(0)
            g,local=model.encode_image(im);g=model.image_post(g);local=model.image_post(local)
            conditioned=model.visual_proj(t.unsqueeze(0),local,local)
            native=(F.normalize(conditioned,dim=-1)[0]*features).sum(-1)*model.logit_scale.exp()
            global_scores=(F.normalize(g,dim=-1).reshape(1,-1)@features.T)[0]*model.logit_scale.exp()
            saved={'native':native,'global':global_scores,'fingerprint':sig}
            if i==0:
                expected,_=compute_similarity_scores_attn_pool(model,[(im,torch.tensor([0]))],features,
                    t,torch.device('cpu'),torch.float32,nullcontext,'inference_with_flair')
                error=float((expected[0]-native).abs().max())
                if error>1e-4:raise RuntimeError('FLAIR official scoring mismatch')
                atomic_json({'max_abs_score_error':error,'provenance':provenance},out/'official_equivalence.json')
            atomic_save(saved,path)
        scores[name]=saved
        if (i+1)%64==0:print(json.dumps({'backend':'flair','images':i+1,'total':len(images)}),flush=True)
    for mode in ['native','global']:
        cases=[]
        for r in records:
            sa,sb,sc=[float(scores[r[k]][mode][text_index[r['query']]]) for k in ['anchor','relevant_change','irrelevant_change']]
            cases.append({'id':r['id'],'scene_group':r['scene_group'],'attribute':r['attribute'],
                          'anchor_margin':sa-sb,'nuisance_margin':sc-sb})
        metrics={}
        for attr in sorted({r['attribute'] for r in cases}):
            rows=[r for r in cases if r['attribute']==attr];n=len(rows)
            metrics[attr]={'cases':n,'anchor_preference':sum(r['anchor_margin']>0 for r in rows)/n,
                'robust_both_preference':sum(min(r['anchor_margin'],r['nuisance_margin'])>0 for r in rows)/n}
        atomic_json({'backend':'flair_'+mode,'dataset_sha256':meta['records_sha256'],'provenance':provenance,
            'metrics':metrics,'cases':cases,'seconds_this_invocation':time.perf_counter()-started,
            'candidate_encoding':'Image token backbone cached; attention pooling repeated for every query' if mode=='native'
                else 'Query-independent global image vector',
            'scope':'Official pretrained inference; no task training. Native is a query-dependent reranking comparator, not fixed-vector ANN.'},
            Path('artifacts/results/v5')/f'evidence_flair_{mode}.json')
        print(json.dumps({'mode':mode,'metrics':metrics}),flush=True)


if __name__=='__main__':main()
