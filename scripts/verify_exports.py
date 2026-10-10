"""Check genuine image inference against cached-head results after export/reload."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from superembedding import OptionEmbeddingHead
from superembedding.omnijev import load_omnijev,OmniJevEmbeddingEncoder


def main():
    root=Path('data/fashion2400')
    record=json.loads((root/'records.jsonl').read_text().splitlines()[0])
    feature=torch.load(Path('runs/fashion2400/omnijev')/f"{record['id']}.pt",weights_only=True)
    results=json.loads(Path('runs/fashion2400/results_omnijev.json').read_text())
    model=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
    output={}
    for arch in ['raw','gated_frozen','gated_tuned','residual','joint_concat','joint_out768','joint_out256']:
        chosen=next(r for r in results['architectures'][arch]['seeds'] if r['seed']==42)
        head,basis,_=OptionEmbeddingHead.load(chosen['checkpoint'])
        with torch.no_grad(): expected=head(feature['u'],feature['zq'])
        encoder=OmniJevEmbeddingEncoder(model,basis)
        encoder.load_head(chosen['checkpoint'])
        actual=encoder.encode({'images':[str(root/record['image'])]})
        ds=(actual.slots.cpu()-expected.slots).abs().max().item()
        dv=(actual.embedding.cpu()-expected.embedding).abs().max().item()
        if max(ds,dv)>1e-5: raise RuntimeError(f'{arch} reload mismatch: {ds}, {dv}')
        output[arch]={'slot_max_abs_diff':ds,'vector_max_abs_diff':dv,'shape':list(actual.embedding.shape)}
    del encoder,model
    torch.cuda.empty_cache()
    final=Path('runs/fashion2400/lora_raw')
    head,basis,source=OptionEmbeddingHead.load(final/'final_embedding_head.pt')
    model=load_omnijev(source['decision_checkpoint'],source['base_model'],max_pixels=65536)
    encoder=OmniJevEmbeddingEncoder(model,basis)
    encoder.load_head(final/'final_embedding_head.pt')
    reference=torch.load(final/'final_reload_reference.pt',weights_only=True)
    actual=encoder.encode({'images':[reference['image']]})
    ds=(actual.slots.cpu()-reference['slots']).abs().max().item()
    dv=(actual.embedding.cpu()-reference['embedding']).abs().max().item()
    if max(ds,dv)>1e-5:raise RuntimeError('Final LoRA export reload mismatch')
    output['lora_final']={'slot_max_abs_diff':ds,'vector_max_abs_diff':dv,'shape':list(actual.embedding.shape)}
    Path('artifacts/results/export_validation.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
