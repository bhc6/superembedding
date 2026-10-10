"""Ensure batching never shares image state or leaks across option branches."""
from types import SimpleNamespace
from pathlib import Path
import sys
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos/OmniJev'))
from mso import branch as BR
from superembedding.batched import encode_prepared

class ImageRecurrence(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(30,12)
    def forward(self,input_ids,past_key_values=None,pixel_values=None,**kwargs):
        h=self.embedding(input_ids).cumsum(1)
        if pixel_values is not None: h=h+pixel_values[:,None,:]
        if past_key_values is not None: h=h+past_key_values.layers[0].state[:,None,:]
        return SimpleNamespace(hidden_states=[h],past_key_values=SimpleNamespace(layers=[SimpleNamespace(state=h[:,-1])]))

def setup():
    torch.manual_seed(42); backbone=ImageRecurrence()
    mso=SimpleNamespace(dev=torch.device('cpu'),model=SimpleNamespace(rope_positions=lambda *a:None),
                        pad_id=0,open_ids=[10],close_ids=[11])
    encoder=SimpleNamespace(_mso=mso,backbone=backbone)
    rows=[[3,10,4,11],[3,10,5,6,11]]
    items=[{'key':('same-layout',),'rows':rows,'prefix':{'input_ids':torch.tensor([[1,2]]),
        'attention_mask':torch.ones(1,2,dtype=torch.long),'pixel_values':torch.randn(1,12)}} for _ in range(4)]
    return encoder,items

def test_image_and_option_states_are_independent():
    encoder,items=setup(); u,q=encode_prepared(encoder,items)
    for i,item in enumerate(items):
        ref=BR.branch_questions(encoder.backbone,item['prefix'],[item['rows']],torch.device('cpu'),[10],[11])[0]
        torch.testing.assert_close(u[i],ref[0],rtol=0,atol=0)
        torch.testing.assert_close(q[i],ref[1],rtol=0,atol=0)
    assert not torch.equal(u[0],u[1])
    order=[3,1,0,2]; reordered=encode_prepared(encoder,[items[i] for i in order])
    torch.testing.assert_close(reordered[0],u[order],rtol=0,atol=0)
    torch.testing.assert_close(reordered[1],q[order],rtol=0,atol=0)

def test_mismatched_layout_rejected():
    encoder,items=setup(); items[-1]['key']=('different-layout',)
    with pytest.raises(ValueError,match='identical token/grid layouts'): encode_prepared(encoder,items)
