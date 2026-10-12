"""Portable inference state for the exact validation-selected general encoder."""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from .checkpoint import atomic_save, fingerprint
from .general import GENERAL_BASIS

FORMAT='superembedding-general-inference-v1'


def export_checkpoint(checkpoint, output):
    source=Path(checkpoint);raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    import io
    saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    config=saved['config']
    if saved.get('fingerprint')!=fingerprint(config):raise ValueError('Checkpoint fingerprint mismatch')
    names=set(saved['model'])
    if not any(n.startswith('backbone.') and 'lora_' in n for n in names):raise ValueError('Missing trained LoRA')
    if not any(n.startswith('embedding_head.') for n in names):raise ValueError('Missing trained embedding head')
    if config['basis']!=list(GENERAL_BASIS.directions):raise ValueError('Unsupported direction basis')
    bundle={'format':FORMAT,'model':saved['model'],'config':deepcopy(config),'fingerprint':saved['fingerprint'],
        'step':saved['step'],'source_checkpoint_sha256':sha,
        'encoder':{'class':'GeneralOmniEncoder','basis_instruction':GENERAL_BASIS.instruction,
                   'max_text_tokens':256,'max_pixels':config['max_pixels'],
                   'output':'L2 normalized concat of four 256d slots'},
        'scope':'Inference weights: ALL trained LoRA and projection tensors. Frozen base and original adapter required. Not an optimizer resume checkpoint.'}
    target=Path(output)
    if target.exists():
        previous=torch.load(target,map_location='cpu',weights_only=True)
        if previous.get('source_checkpoint_sha256')!=sha:raise ValueError('Refuse to replace export from another checkpoint')
        if set(previous.get('model',{}))!=names or any(not torch.equal(previous['model'][n],saved['model'][n]) for n in names):
            raise ValueError('Existing export tensor mismatch')
        if any(previous.get(k)!=v for k,v in bundle.items() if k!='model'):raise ValueError('Existing export metadata mismatch')
    else:atomic_save(bundle,target)
    return {'checkpoint_sha256':sha,'export_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'step':saved['step'],'trained_tensors':len(names),'export':str(target)}


def validate_bundle(saved):
    if saved.get('format')!=FORMAT:raise ValueError('Use a complete general inference bundle, not a standalone head')
    expected={'class':'GeneralOmniEncoder','basis_instruction':GENERAL_BASIS.instruction,
              'max_text_tokens':256,'max_pixels':saved['config']['max_pixels'],
              'output':'L2 normalized concat of four 256d slots'}
    if saved['encoder']!=expected:raise ValueError('General encoder/preprocessing contract mismatch')
    if saved.get('fingerprint')!=fingerprint(saved['config']):raise ValueError('Bundle fingerprint mismatch')


def load_bundle_encoder(path):
    from .omnijev import load_omnijev
    from .general import GeneralOmniEncoder
    from .general_inference import load_general_state
    saved=torch.load(path,map_location='cpu',weights_only=True);validate_bundle(saved)
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=saved['config']['max_pixels'])
    model=GeneralOmniEncoder(mso,GENERAL_BASIS,embed_dim=256,mode='raw',tune_lora=False).eval()
    load_general_state(model,saved)
    return model,saved
