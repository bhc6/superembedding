"""Export and verify full LoRA+head inference state; no test-based selection."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--image',required=True)
    args=ap.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from superembedding.checkpoint import atomic_json
    from superembedding.general_export import export_checkpoint,load_bundle_encoder
    from superembedding.general import GENERAL_BASIS,GeneralOmniEncoder
    from superembedding.general_inference import load_general_state
    from superembedding.omnijev import load_omnijev
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    info=export_checkpoint(args.checkpoint,args.output)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()!=info['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed during export; use an immutable snapshot')
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=saved['config']['max_pixels'])
    original=GeneralOmniEncoder(mso,GENERAL_BASIS,embed_dim=256,mode='raw',tune_lora=False).eval()
    load_general_state(original,saved)
    objects=[{'text':'Find the visual evidence for the described objects.'},
             {'images':[args.image]}, {'text':'Which objects and written details are visible?','images':[args.image]}]
    with torch.no_grad():expected=[original(o).embedding.cpu() for o in objects]
    del original,mso;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    model,bundle=load_bundle_encoder(args.output)
    exact=set(saved['model'])==set(bundle['model']) and all(torch.equal(t,bundle['model'][n]) for n,t in saved['model'].items())
    if not exact:raise ValueError('Export did not preserve ALL trained tensors')
    with torch.no_grad():actual=[model(o).embedding.cpu() for o in objects]
    diffs=[float((a-b).abs().max()) for a,b in zip(expected,actual)]
    if max(diffs)>1e-6:raise ValueError('Reloaded inference vectors differ from formal checkpoint')
    result={**info,'passed':True,'all_trained_tensors_equal':exact,'input_types':['text','image','image+text'],
        'embedding_max_abs_errors':diffs,'device':args.device,'image_sha256':hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
        'scope':'Exact checkpoint-to-bundle reload on real inputs; not a retrieval metric or quality claim.'}
    atomic_json(result,Path(args.output).with_suffix('.audit.json'));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
