"""Reject incomplete LoRA/head states before frozen V5 evaluation."""
import hashlib
from pathlib import Path
from .checkpoint import fingerprint


def load_general_state(model, saved, verify_sources=True):
    config=saved['config']
    if 'fingerprint' in saved and saved['fingerprint']!=fingerprint(config):
        raise ValueError('Checkpoint configuration fingerprint mismatch')
    if config['basis']!=list(model.basis.directions):raise ValueError('Checkpoint basis mismatch')
    if config['projection']!='raw-shared-256-no-output-projection':
        raise ValueError('Unsupported general projection architecture')
    if model.embedding_head.mode!='raw' or model.embedding_head.embed_dim!=256:
        raise ValueError('Evaluation encoder architecture mismatch')
    expected={name for name,_ in model.named_parameters()
              if name.startswith('embedding_head.') or (name.startswith('backbone.') and 'lora_' in name)}
    actual=set(saved['model'])
    if actual!=expected:
        raise ValueError(f'Incomplete/incompatible checkpoint: missing {len(expected-actual)}, extra {len(actual-expected)}')
    if verify_sources:
        for label,path in [('adapter','models/OmniJev-0.8B-v1.1/adapter_model.safetensors'),
                           ('base','models/Qwen3.5-0.8B/source_revision.json')]:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=config[label+'_sha256']:
                raise ValueError('Frozen source mismatch: '+label)
        for path in ['superembedding/general.py','superembedding/heads.py']:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=config['implementation_sha256'][path]:
                raise ValueError('Encoder implementation mismatch: '+path)
    result=model.load_state_dict(saved['model'],strict=False)
    if result.unexpected_keys or expected.intersection(result.missing_keys):
        raise ValueError('Trained tensor load mismatch')
    return {'trained_tensors':len(expected),'step':saved['step'],'config_fingerprint':fingerprint(config)}
