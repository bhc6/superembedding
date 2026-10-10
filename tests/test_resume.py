import random
import pytest
import torch
from superembedding.checkpoint import save_training, restore_training
from superembedding.checkpoint import atomic_save
from superembedding import OptionEmbeddingHead, AttributeBasis


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_resume_replays_optimizer_and_rng_exactly(tmp_path,device):
    torch.manual_seed(123)
    model=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.Dropout(.2),torch.nn.Linear(8,1)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    rng=random.Random(42)
    config={'dataset':'hash','steps':6}
    def step():
        x=torch.randn(10,8,device=device)*rng.random()
        loss=model(x).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
        return loss.detach().cpu()
    for _ in range(3): step()
    save_training(tmp_path/'last.pt',model,opt,3,config,rng,{'best':.5})
    losses=[step() for _ in range(3)]
    expected={k:v.clone() for k,v in model.state_dict().items()}
    restored,extra=restore_training(tmp_path/'last.pt',model,opt,config,rng)
    assert restored==3 and extra['best']==.5
    for loss in losses: torch.testing.assert_close(step(),loss,rtol=0,atol=0)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='mismatch'):
        restore_training(tmp_path/'last.pt',model,opt,{'dataset':'other'},rng)


def test_residual_output_projection_preserves_slots_and_serialization(tmp_path):
    torch.manual_seed(8)
    head=OptionEmbeddingHead(12,16,8,'residual',True,num_slots=3,output_dim=10)
    u,zq=torch.randn(5,3,12),torch.randn(5,12)
    first=head(u,zq)
    assert first.embedding.shape==(5,10)
    (first.embedding*torch.randn_like(first.embedding)).sum().backward()
    for layer in [head.output_projection,head.residual_projection[1],head.projection[1],head.opt[0],head.que[0]]:
        assert layer.weight.grad.abs().sum()>0
    with torch.no_grad(): head.output_projection.weight.add_(.2)
    torch.testing.assert_close(head(u,zq).slots,first.slots)
    basis=AttributeBasis(('color','type','audience'))
    head.save(tmp_path/'head.pt',basis)
    restored,_,_=OptionEmbeddingHead.load(tmp_path/'head.pt')
    torch.testing.assert_close(restored(u,zq).embedding,head(u,zq).embedding)
    with pytest.raises(ValueError,match='slots'): head(u[:,:2],zq)


def test_partial_snapshot_retains_frozen_backbone_and_restores_adapters(tmp_path):
    torch.manual_seed(9)
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2))
    model[0].requires_grad_(False)
    opt=torch.optim.AdamW(model[1].parameters(),lr=.01)
    frozen=model[0].weight.clone()
    expected=model[1].weight.clone()
    rng=random.Random(4)
    save_training(tmp_path/'last.pt',model,opt,0,{},rng,trainable_only=True)
    with torch.no_grad(): model[1].weight.add_(1)
    restore_training(tmp_path/'last.pt',model,opt,{},rng)
    torch.testing.assert_close(model[0].weight,frozen)
    torch.testing.assert_close(model[1].weight,expected)


def test_failed_atomic_replacement_preserves_last_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'last.pt'
    atomic_save({'step':10},path)
    def fail(*args): raise OSError('simulated interruption before commit')
    monkeypatch.setattr('superembedding.checkpoint.os.replace',fail)
    with pytest.raises(OSError): atomic_save({'step':20},path)
    assert torch.load(path,weights_only=True)['step']==10
