"""Fresh training-image forwards vs cached features and reloaded released heads."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from superembedding import OptionEmbeddingHead
from superembedding.baselines import DATASETS,MODELS,read_data,load_packed
from superembedding.checkpoint import atomic_json
from scripts.encode_baseline_image import encode_features
from scripts.extract_baselines import local_model

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--model-cache',default='models/v4');a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    seal=json.loads(Path('artifacts/results/v4/selection_sealed.json').read_text());checks=[]
    for b,spec in MODELS.items():
        loaded=local_model(b,a.model_cache) if 'repo' in spec else None
        for d in DATASETS:
            root,meta,records=read_data(d);cache=Path(a.features)/d/b;prov=json.loads((cache/'provenance.json').read_text())
            u,z=load_packed(cache,records,prov)
            ids=sorted([i for i,r in enumerate(records) if r['split']=='train'],key=lambda i:records[i]['train_rank'])[:2]
            item=next(i for i in seal['selected'] if i['dataset']==d and i['backend']==b and i['scale']==100 and i['seed']==42)
            head,basis,_=OptionEmbeddingHead.load(Path(item['path'])/'best_head.pt');head.eval()
            fresh,q=encode_features([root/records[i]['image'] for i in ids],b,basis,a.model_cache,loaded)
            cosine=F.cosine_similarity(u[ids],fresh,dim=-1).min().item()
            old=head(u[ids],z[ids]);new=head(fresh,q)
            delta=(old.slots-new.slots).abs().max().item();headcos=F.cosine_similarity(old.slots,new.slots,dim=-1).min().item()
            # Remote aliases may drift: preserve observations and explicitly fail on material changes.
            assert cosine>.9999 and headcos>.9999,(d,b,cosine,headcos)
            if 'repo' in spec:assert delta<1e-4,(d,b,delta)
            check=dict(dataset=d,backend=b,train_ids=[records[i]['id'] for i in ids],
                min_feature_cosine=cosine,min_head_cosine=headcos,max_head_abs_difference=delta)
            checks.append(check);print(json.dumps(check),flush=True)
        del loaded;torch.cuda.empty_cache()
    atomic_json({'checks':checks},'artifacts/results/v4/fresh_inference_verification.json')

if __name__=='__main__':main()
