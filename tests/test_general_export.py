import pytest
import torch
from superembedding.checkpoint import atomic_save,fingerprint
from superembedding.general import GENERAL_BASIS
from superembedding.general_export import export_checkpoint,validate_bundle


def checkpoint(path,step=10):
    config={'basis':list(GENERAL_BASIS.directions),'max_pixels':65536}
    obj={'config':config,'fingerprint':fingerprint(config),'step':step,
         'model':{'backbone.lora_A.weight':torch.randn(2,2),'embedding_head.projection.weight':torch.randn(2,2)},
         'optimizer':{'not_for_inference':True}}
    atomic_save(obj,path);return obj


def test_export_preserves_selected_step_and_lora_and_head(tmp_path):
    source=tmp_path/'best.pt';target=tmp_path/'model.pt';obj=checkpoint(source)
    export_checkpoint(source,target);saved=torch.load(target,weights_only=True)
    validate_bundle(saved)
    assert saved['step']==10 and 'optimizer' not in saved
    assert all(torch.equal(t,saved['model'][k]) for k,t in obj['model'].items())


def test_export_refuses_to_overwrite_best_with_last(tmp_path):
    source=tmp_path/'best.pt';target=tmp_path/'model.pt';checkpoint(source)
    export_checkpoint(source,target);checkpoint(source,step=20)
    with pytest.raises(ValueError,match='another checkpoint'):export_checkpoint(source,target)


def test_standalone_head_is_not_a_general_model():
    with pytest.raises(ValueError,match='standalone head'):validate_bundle({'format_version':1,'state_dict':{}})


def test_modified_input_contract_is_rejected(tmp_path):
    source=tmp_path/'best.pt';target=tmp_path/'model.pt';checkpoint(source);export_checkpoint(source,target)
    saved=torch.load(target,weights_only=True);saved['encoder']['max_text_tokens']=128
    with pytest.raises(ValueError,match='contract'):validate_bundle(saved)


def test_missing_lora_refuses_export(tmp_path):
    source=tmp_path/'best.pt';obj=checkpoint(source);del obj['model']['backbone.lora_A.weight'];atomic_save(obj,source)
    with pytest.raises(ValueError,match='LoRA'):export_checkpoint(source,tmp_path/'model.pt')
