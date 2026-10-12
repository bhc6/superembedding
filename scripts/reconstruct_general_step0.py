"""Reconstruct, never claim to recover, overwritten initial inference weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--parent',default='D:/DEMO/superembedding-storage/superembedding/v5/general_stage1/last.pt')
    ap.add_argument('--config',default='D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/config.json')
    ap.add_argument('--output',default='D:/DEMO/superembedding-storage/superembedding/v5/step0_reconstructed_cpu')
    args=ap.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['MSO_FLA']='0';os.environ['MSO_BRANCH_CHUNK']='4'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from superembedding.checkpoint import atomic_json,atomic_save,fingerprint
    from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS
    from superembedding.general_export import export_checkpoint
    from superembedding.omnijev import load_omnijev
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists():raise ValueError('Immutable reconstruction exists; inspect instead of overwriting')
    if parent['config']['seed']!=config['seed']:raise ValueError('Initialization seed mismatch')
    for key in ['adapter_sha256','base_sha256','basis','projection','torch','transformers']:
        if parent['config'][key]!=config[key]:raise ValueError('Parent model/config mismatch: '+key)
    for file in ['superembedding/general.py','superembedding/heads.py']:
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=parent['config']['implementation_sha256'][file]:
            raise ValueError('Historical constructor implementation changed')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=config['max_pixels'])
    model=GeneralOmniEncoder(mso,GENERAL_BASIS,embed_dim=256,mode='raw',tune_lora=True).eval()
    names={n for n,p in model.named_parameters() if p.requires_grad}
    state={n:p.detach().cpu().clone() for n,p in model.state_dict().items() if n in names}
    cpu_rng_equal=torch.equal(torch.get_rng_state(),parent['rng_torch'])
    differences={n:float((v-parent['model'][n]).abs().max()) for n,v in state.items() if n.startswith('embedding_head.')}
    # Eval-mode backbone/replay uses no CPU stochastic training ops. Matching
    # RNG consumption and small head deltas are evidence, not a saved step-0 hash.
    audit={'status':'reconstructed_not_original_saved_checkpoint','seed':config['seed'],'device':args.device,
           'parent_step':parent['step'],'parent_sha256':hashlib.sha256(Path(args.parent).read_bytes()).hexdigest(),
           'cpu_rng_matches_parent':cpu_rng_equal,'head_delta_to_parent':differences,
           'head_state_sha256':fingerprint({n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in state.items() if n.startswith('embedding_head.')}),
           'limitation':'Original step-0 checkpoint was overwritten. Same seed/constructor reconstruction cannot be certified bit-identical without an original hash. CPU/GPU loading can consume different RNG.'}
    atomic_json(audit,out/'reconstruction.json')
    if not cpu_rng_equal or max(differences.values())>.01:
        raise RuntimeError('Initialization replay consistency failed; do not use this as Step 0')
    atomic_save({'model':state,'step':0,'config':config,'fingerprint':fingerprint(config),
                 'reconstruction':audit},out/'checkpoint.pt')
    export_checkpoint(out/'checkpoint.pt',out/'general_model.pt')
    atomic_json(audit,Path('artifacts/results/v5')/f'step0_reconstruction_{args.device}.json')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
