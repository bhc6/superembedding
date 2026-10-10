"""Export final trained weights separately from the validation-selected adapter."""
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MSO_FLA','0');os.environ.setdefault('MSO_BRANCH_CHUNK','4')
import torch
from superembedding import OptionEmbeddingHead
from superembedding.omnijev import load_omnijev,OmniJevEmbeddingEncoder
from superembedding.checkpoint import atomic_save


def main():
    root=Path('runs/fashion2400/lora_raw').resolve()
    report=json.loads((root/'training.json').read_text())
    config=report['config']
    head,basis,_=OptionEmbeddingHead.load(config['head'])
    model=load_omnijev(config['adapter'],config['base'],max_pixels=config['max_pixels'])
    encoder=OmniJevEmbeddingEncoder(model,basis,tune_lora=True)
    encoder.load_head(config['head'])
    initial={n:p.detach().cpu().clone() for n,p in encoder.named_parameters() if p.requires_grad and 'lora_' in n}
    last=torch.load(root/'last.pt',weights_only=True,map_location='cpu')
    trainable={n for n,p in encoder.named_parameters() if p.requires_grad}
    if trainable != set(last['model']):raise ValueError('Partial checkpoint parameter mismatch')
    encoder.load_state_dict(last['model'],strict=False)
    deltas=[(last['model'][n]-value).abs().max().item() for n,value in initial.items()]
    adapter=root/'final_adapter'
    encoder.backbone.save_pretrained(adapter,safe_serialization=True)
    for filename in ['tokenizer.json','tokenizer_config.json','processor_config.json','chat_template.jinja',
                     'new_tok_emb.pt','head.pt','ord.pt','LICENSE']:
        p=Path(config['adapter'])/filename
        if p.exists():shutil.copy2(p,adapter/filename)
    metadata=json.loads((root/'adapter/head_meta.json').read_text())
    metadata.update(embedding_finetuned=True,selected_training_step=last['step'],selection='final update, not validation selected')
    (adapter/'head_meta.json').write_text(json.dumps(metadata,indent=2))
    selected=json.loads((root/'adapter/head_meta.json').read_text())
    selected.update(embedding_finetuned=report['best_step']>0,selected_training_step=report['best_step'],selection='best validation attribute mAP')
    (root/'adapter/head_meta.json').write_text(json.dumps(selected,indent=2))
    encoder.source={'decision_checkpoint':str(adapter),'base_model':str(Path(config['base']).resolve())}
    encoder.save_head(root/'final_embedding_head.pt')
    reference=torch.load(root/'reload_reference.pt',weights_only=True)
    output=encoder.encode({'images':[reference['image']]})
    atomic_save({'image':reference['image'],'slots':output.slots.cpu(),'embedding':output.embedding.cpu()},root/'final_reload_reference.pt')
    result={'final_step':last['step'],'validation_selected_step':report['best_step'],
            'final_lora_max_abs_weight_change':max(deltas),'changed_lora_tensors':sum(d>0 for d in deltas),
            'selected_is_unmodified_initialization':report['best_step']==0,
            'final_adapter':str(adapter),'final_head':str(root/'final_embedding_head.pt')}
    (root/'export_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
