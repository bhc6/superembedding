from types import SimpleNamespace
import pytest
import torch
from superembedding.general_inference import load_general_state
from superembedding.checkpoint import fingerprint


def fixture():
    model=torch.nn.Module();model.backbone=torch.nn.Module()
    model.backbone.lora_A=torch.nn.Linear(2,2,bias=False)
    model.backbone.dense=torch.nn.Linear(2,2)
    model.embedding_head=torch.nn.Linear(2,2)
    model.embedding_head.mode='raw';model.embedding_head.embed_dim=256
    model.basis=SimpleNamespace(directions=['test'])
    config={'basis':['test'],'projection':'raw-shared-256-no-output-projection'}
    state={k:v.clone() for k,v in model.state_dict().items() if 'dense' not in k}
    return model,{'model':state,'config':config,'fingerprint':fingerprint(config),'step':60}


def test_partial_state_loads_all_trained_tensors_without_dense_weights():
    model,saved=fixture()
    assert load_general_state(model,saved,verify_sources=False)['trained_tensors']==3


@pytest.mark.parametrize('key',['embedding_head.weight','backbone.lora_A.weight'])
def test_missing_head_or_lora_is_rejected_before_mutation(key):
    model,saved=fixture();del saved['model'][key]
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError,match='Incomplete'):load_general_state(model,saved,verify_sources=False)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_changed_basis_is_rejected():
    model,saved=fixture();saved['config']['basis']=['different']
    saved['fingerprint']=fingerprint(saved['config'])
    with pytest.raises(ValueError,match='basis'):load_general_state(model,saved,verify_sources=False)
