"""Frozen general checkpoint transfer to Urban1k and paired evidence diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from torch.nn import functional as F
from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS
from superembedding.omnijev import load_omnijev
from superembedding.checkpoint import atomic_save,atomic_json,fingerprint
from superembedding.general_inference import load_general_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=ap.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    tag='general_'+hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()[:12] if args.checkpoint else 'omnijev_raw'
    if args.device=='cpu':tag+='_cpu'
    output=Path('runs/v5/transfer')/tag;output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2 if args.device=='cpu' else 4)
    if args.checkpoint:
        from superembedding.general_export import load_bundle_encoder
        model,saved=load_bundle_encoder(args.checkpoint)
    else:
        mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
        model=GeneralOmniEncoder(mso,GENERAL_BASIS,mode='raw',tune_lora=False).eval()
    @torch.no_grad()
    def encode(obj):
        sig=fingerprint({'object':obj,'checkpoint':tag,'basis':list(GENERAL_BASIS.directions)})
        path=output/(sig+'.pt')
        if path.exists():return torch.load(path,weights_only=True)['vector']
        if args.checkpoint:v=model(obj).embedding.cpu()
        else:
            u,_=model.extract_features(obj);v=F.normalize(F.normalize(u.float(),dim=-1).flatten(),dim=-1).cpu()
        atomic_save({'vector':v,'fingerprint':sig},path);return v
    urban=Path('C:/Users/123/.cache/superembedding/v5/urban1k')
    rows=json.loads((urban/'annotations/annotations.json').read_text(encoding='utf-8'))['annotations']
    valid=[r for r in rows if r['caption'].strip()]
    q=[];d=[]
    for i,r in enumerate(valid,1):
        q.append(encode({'text':r['caption']}));d.append(encode({'images':[str(urban/'images'/r['image'])]}))
        if i%100==0:print(json.dumps({'backend':tag,'urban_pairs':i,'total':len(valid)}),flush=True)
    scores=torch.stack(d)@torch.stack(q).T;n=len(valid);metrics={}
    for direction,s in [('image_to_text',scores),('text_to_image',scores.T)]:
        ranks=s.argsort(-1,descending=True)
        for k in [1,5,10]:metrics[f'{direction}_R@{k}']=(ranks[:,:k]==torch.arange(n)[:,None]).any(-1).float().mean().item()
    root=Path('data/evidence_diagnostic_v1');records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    diagnostic_sha256=hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest()
    if diagnostic_sha256!=json.loads((root/'metadata.json').read_text())['records_sha256']:
        raise ValueError('Diagnostic manifest changed')
    cases=[]
    for r in records:
        a,b,c=[encode({'images':[str(root/r[k])]}) for k in ['anchor','relevant_change','irrelevant_change']]
        query=encode({'text':r['query']});sa,sb,sc=[float(query@v) for v in [a,b,c]]
        cases.append({'id':r['id'],'scene_group':r['scene_group'],'attribute':r['attribute'],
                      'anchor_margin':sa-sb,'nuisance_margin':sc-sb,
                      'target_change_distance':float(1-a@b),'nuisance_distance':float(1-a@c)})
    diagnostic={}
    for attribute in sorted({r['attribute'] for r in cases}):
        group=[r for r in cases if r['attribute']==attribute]
        diagnostic[attribute]={'anchor_preference':sum(r['anchor_margin']>0 for r in group)/len(group),
                              'robust_both_preference':sum(min(r['anchor_margin'],r['nuisance_margin'])>0 for r in group)/len(group)}
    result={'backend':tag,'checkpoint':args.checkpoint,'urban1k_common_999':metrics,'evidence_diagnostic':diagnostic,
            'cases':cases,'scope':'Frozen transfer only; never used for checkpoint selection',
            'trained_projection':bool(args.checkpoint),'max_pixels':65536,'max_text_tokens':256,
            'device':args.device,'dtype':'float32' if args.device=='cpu' else 'bfloat16'}
    atomic_json(result,Path('artifacts/results/v5')/f'transfer_{tag}.json');print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    atomic_json({'backend':tag,'dataset_sha256':diagnostic_sha256,'metrics':diagnostic,'cases':cases,
        'checkpoint':args.checkpoint,'scope':result['scope'],'trained_projection':bool(args.checkpoint),
        'candidate_encoding':'Query-independent fixed four-direction vector'},
        Path('artifacts/results/v5')/f'evidence_{tag}.json')

if __name__=='__main__':main()
