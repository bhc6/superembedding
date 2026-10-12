"""Apply the explicitly reconstructed initial head to pinned raw CPU features."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.checkpoint import atomic_json,fingerprint
from superembedding.heads import OptionEmbeddingHead
from superembedding.validation_protocol import load_protocol,score_vectors


def main():
    torch.set_num_threads(2)
    root=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    model_root=root/'step0_reconstructed_cpu'
    audit=json.loads((model_root/'reconstruction.json').read_text(encoding='utf-8'))
    if not audit['cpu_rng_matches_parent']:raise ValueError('Reconstruction evidence failed')
    saved=torch.load(model_root/'checkpoint.pt',map_location='cpu',weights_only=True)
    if saved['step']!=0:raise ValueError('Not initial weights')
    protocol,items,_=load_protocol(root/'data_v51')
    features=root/'general_ablation_features'
    prov=json.loads((features/'provenance.json').read_text(encoding='utf-8'))
    sig=fingerprint(prov)
    if prov['data_sha256']!=protocol['data_sha256']:raise ValueError('Data mismatch')
    if prov['adapter_sha256']!=saved['config']['adapter_sha256'] or prov['base_sha256']!=saved['config']['base_sha256']:
        raise ValueError('Frozen backbone mismatch')
    if prov['basis']!=saved['config']['basis']:raise ValueError('Basis mismatch')
    indices={s:i for i,s in enumerate(prov['ids'])}
    head=OptionEmbeddingHead(1024,embed_dim=256,mode='raw').eval()
    head.load_state_dict({k.removeprefix('embedding_head.'):v for k,v in saved['model'].items() if k.startswith('embedding_head.')},strict=True)
    q=[];d=[]
    with torch.no_grad():
        for row in items:
            feature=torch.load(features/'shards'/f"{indices[row['id']]:05d}.pt",weights_only=True,map_location='cpu')
            if feature['fingerprint']!=sig or feature['id']!=row['id'] or feature['key']!=row['key']:
                raise ValueError('Feature identity mismatch')
            q.append(head(feature['q']).embedding);d.append(head(feature['d']).embedding)
    result={'provenance':{'protocol':protocol,'reconstruction':audit,'feature_provenance':prov,
                          'checkpoint_sha256':hashlib.sha256((model_root/'checkpoint.pt').read_bytes()).hexdigest()},
            'full':score_vectors(items,torch.stack(q),torch.stack(d)),
            'limitation':'Reconstructed initial head, NOT original saved Step 0. CPU float32 features; training validation is CUDA bfloat16. Do not present as a precision-controlled causal training delta.'}
    atomic_json(result,'artifacts/results/v5/validation_controls/step0_reconstructed_cpu.json')
    print(json.dumps({'reconstructed_step0_macro_r1':result['full']['macro_r1']}),flush=True)


if __name__=='__main__':main()
