"""Pinned mainstream image embeddings and resumable, credential-free feature banks."""
from pathlib import Path
import hashlib
import json
import torch
from .checkpoint import fingerprint, atomic_save, atomic_json

DATASETS = ('fashion_expanded', 'cub_attributes', 'dtd')
MODELS = {
    'clip_b32': dict(repo='openai/clip-vit-base-patch32', revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', width=512, family='CLIP contrastive dual encoder'),
    'siglip2_b16': dict(repo='google/siglip2-base-patch16-224', revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2', width=768, family='SigLIP2 sigmoid image-text encoder'),
    'dinov2_small': dict(repo='facebook/dinov2-small', revision='ed25f3a31f01632728cabb09d1542f84ab7b0056', width=384, family='DINOv2 visual self-supervision'),
    'qwen3_vl': dict(model='qwen3-vl-embedding', width=2560, family='Qwen VL embedding API', parameters={'dimension':2560}),
    'qwen3_vl_instruct': dict(model='qwen3-vl-embedding', width=2560, family='Qwen VL embedding API, fixed attribute instructions', parameters={'dimension':2560}, conditioned=True),
    'tongyi_plus': dict(model='tongyi-embedding-vision-plus-2026-03-06', width=1152, family='Tongyi vision embedding API, dated snapshot', parameters={'dimension':1152,'res_level':1}),
}
ENDPOINT = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding'

def read_data(dataset):
    root = Path('data') / dataset
    meta = json.loads((root/'metadata.json').read_text())
    records = [json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    return root, meta, records

def instruction(direction):
    return f'Represent the image for retrieval by {direction}. Focus on this attribute and ignore other attributes.'

def provenance(dataset, backend):
    _, meta, records = read_data(dataset)
    spec = MODELS[backend]
    return dict(protocol='v4', backend=backend, spec=spec,
        basis={'directions':meta['directions'],'instruction':meta['instruction']},
        record_fingerprint=fingerprint(records),
        conditioning='fixed attribute instruct' if spec.get('conditioned') else 'global image; repeated identical slots',
        prompts=[instruction(d) for d in meta['directions']] if spec.get('conditioned') else [],
        preprocessing='original image bytes; provider preprocessing' if 'model' in spec else 'official Hugging Face image processor; float32 forward',
        endpoint=ENDPOINT if 'model' in spec else None,
        provider_revision='alias not immutable; cached vectors define this run' if backend.startswith('qwen3') else spec.get('revision',spec.get('model')))

def initialize(out, prov):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != prov:
        raise ValueError('Feature provenance mismatch')
    atomic_json(prov,path)

def image_bytes(root,record):
    import io
    from PIL import Image
    data=(root/record['image']).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image=image.convert('RGB')
        digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    if digest!=record['image_sha256']:
        raise ValueError('Image hash changed: '+record['id'])
    return data

def validate_vectors(vectors, count, width):
    if vectors.shape!=(count,width) or not torch.isfinite(vectors).all() or (vectors.norm(dim=-1)<1e-8).any():
        raise ValueError('Invalid embedding shape, finite values, or norm')

def pack(out, records, spec, directions, batch_size):
    out=Path(out); values=[]
    groups=range(len(directions)) if spec.get('conditioned') else [0]
    for group in groups:
        batches=[]
        for offset in range(0,len(records),batch_size):
            r=records[offset:offset+batch_size]
            shard=torch.load(out/'shards'/f'{group}_{offset:06d}.pt',weights_only=True)
            if shard['ids']!=[x['id'] for x in r] or shard['hashes']!=[x['image_sha256'] for x in r]:
                raise ValueError('Shard record mismatch')
            validate_vectors(shard['vectors'],len(r),spec['width'])
            batches.append(shard['vectors'])
        values.append(torch.cat(batches))
    packed={'ids':[r['id'] for r in records],'hashes':[r['image_sha256'] for r in records],
        'provenance_fingerprint':fingerprint(json.loads((out/'provenance.json').read_text())),
        'vectors':torch.stack(values,1) if spec.get('conditioned') else values[0]}
    atomic_save(packed,out/'features.pt')
    return packed

def load_packed(cache, records, prov):
    p=torch.load(Path(cache)/'features.pt',weights_only=True,map_location='cpu')
    if p['ids']!=[r['id'] for r in records] or p['hashes']!=[r['image_sha256'] for r in records]:
        raise ValueError('Packed feature record mismatch')
    if p['provenance_fingerprint']!=fingerprint(prov):
        raise ValueError('Packed feature provenance mismatch')
    v=p['vectors'].float()
    expected=(len(records),len(prov['basis']['directions']),prov['spec']['width']) if prov['spec'].get('conditioned') else (len(records),prov['spec']['width'])
    if tuple(v.shape)!=expected:
        raise ValueError('Packed feature dimensions mismatch')
    validate_vectors(v.reshape(-1,v.shape[-1]),v.numel()//v.shape[-1],prov['spec']['width'])
    if v.ndim==2:
        return v[:,None].expand(-1,len(prov['basis']['directions']),-1),v
    return v,v.mean(1)
