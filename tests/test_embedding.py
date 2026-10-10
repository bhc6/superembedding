from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repos" / "OmniJev"))
from mso.head import OptionScorer
from superembedding import AttributeBasis, OptionEmbeddingHead, attribute_triplet_loss
from superembedding.omnijev import OmniJevEmbeddingEncoder


def make_head(mode="gated", train_gate=False):
    torch.manual_seed(3)
    original = OptionScorer(12, d_hidden=16)
    return original, OptionEmbeddingHead.from_decision_head(original, 8, mode, train_gate)


def test_gated_readout_copies_weights_and_matches_original_interaction():
    original, head = make_head()
    u, zq = torch.randn(2, 3, 12), torch.randn(2, 12)
    expected_hidden = original.opt(u) * original.que(zq).tanh().unsqueeze(1)
    expected = torch.nn.functional.normalize(head.projection(expected_hidden), dim=-1)
    out = head(u, zq)
    torch.testing.assert_close(out.slots, expected)
    assert out.embedding.shape == (2, 24)
    torch.testing.assert_close(out.slots.norm(dim=-1), torch.ones(2, 3))
    torch.testing.assert_close(out.embedding.norm(dim=-1), torch.ones(2))
    assert head.opt[0].weight.data_ptr() != original.opt[0].weight.data_ptr()


@pytest.mark.parametrize("mode", ["gated", "raw"])
def test_slots_are_independent_of_other_options_and_preserve_order(mode):
    _, head = make_head(mode)
    u, zq = torch.randn(4, 12), torch.randn(12)
    full = head(u, zq)
    torch.testing.assert_close(head(u[[2, 0]], zq).slots, full.slots[[2, 0]])
    torch.testing.assert_close(head(u[:1], zq).slots[0], full.slots[0])


@pytest.mark.parametrize("train_gate", [False, True])
def test_gradients_reach_projection_and_only_enabled_gate(train_gate):
    original, head = make_head(train_gate=train_gate)
    before = {k: v.clone() for k, v in original.state_dict().items()}
    out = head(torch.randn(3, 12), torch.randn(12))
    (out.slots * torch.randn_like(out.slots)).sum().backward()
    assert head.projection[1].weight.grad.abs().sum() > 0
    for module in (head.opt, head.que):
        assert (module[0].weight.grad is not None) == train_gate
    assert all(p.grad is None for p in original.parameters())
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_checkpoint_roundtrip_and_basis_mismatch(tmp_path):
    _, head = make_head(train_gate=True)
    basis = AttributeBasis(("color", "shape"))
    path = tmp_path / "head.pt"
    head.save(path, basis, {"adapter_revision": "test"})
    restored, restored_basis, source = OptionEmbeddingHead.load(path, basis)
    u, zq = torch.randn(2, 12), torch.randn(12)
    torch.testing.assert_close(head(u, zq).embedding, restored(u, zq).embedding)
    assert restored_basis == basis and source['adapter_revision'] == 'test'
    assert restored.opt[0].weight.requires_grad
    with pytest.raises(ValueError, match="order or instruction"):
        OptionEmbeddingHead.load(path, AttributeBasis(("shape", "color")))

def test_encoder_accepts_provenance_metadata_but_rejects_wrong_backend(tmp_path):
    backbone=nn.Linear(12,12); basis=AttributeBasis(('color','shape'))
    source={'decision_checkpoint':'adapter','base_model':'base'}
    mso=SimpleNamespace(branch=True,model=SimpleNamespace(backbone=backbone,head=OptionScorer(12,16)),
                        _superembedding_source=source)
    encoder=OmniJevEmbeddingEncoder(mso,basis,embed_dim=8,mode='raw')
    path=tmp_path/'head.pt'
    encoder.embedding_head.save(path,basis,{**source,'feature_backend':'omnijev','research_protocol':'v3'})
    encoder.load_head(path)
    encoder.embedding_head.save(path,basis,{**source,'feature_backend':'base','research_protocol':'v3'})
    with pytest.raises(ValueError,match='matching feature extractor'): encoder.load_head(path,allow_source_relocation=True)
    encoder.embedding_head.save(path,basis,{**source,'base_model':'wrong'})
    with pytest.raises(ValueError,match='source references'): encoder.load_head(path)


def test_loss_only_supervises_selected_attribute():
    torch.manual_seed(1)
    a, p, n = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(3)]
    loss = attribute_triplet_loss(a, p, n, torch.tensor([0, 2]), margin=2)
    loss.backward()
    assert a.grad[0, 0].abs().sum() > 0 and a.grad[1, 2].abs().sum() > 0
    assert a.grad[:, 1].abs().sum() == 0
    assert a.grad[0, 2].abs().sum() == 0


class ToyBackbone(nn.Module):
    """A cache-capable recurrence to exercise the real branch.py plumbing."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30, 12)
        self.calls = 0

    def forward(self, input_ids, past_key_values=None, **kwargs):
        self.calls += 1
        h = self.embedding(input_ids).cumsum(dim=1)
        if past_key_values is not None:
            h = h + past_key_values.layers[0].state[:, None]
        cache = SimpleNamespace(layers=[SimpleNamespace(state=h[:, -1])])
        return SimpleNamespace(past_key_values=cache, hidden_states=[h])


def test_real_branch_plumbing_with_frozen_toy_backbone():
    backbone = ToyBackbone()
    decision = OptionScorer(12, 16)
    model = SimpleNamespace(backbone=backbone, head=decision, rope_positions=lambda *args: None)
    basis = AttributeBasis(("color", "shape"))
    def encode_many(*args):
        return [["color", "shape"]], [[{}, {}]], [[1, 2, 3, 10, 4, 11, 10, 5, 11]], [([3, 6], [5, 8])], {}
    mso = SimpleNamespace(branch=True, model=model, dev=torch.device('cpu'),
                          open_ids=[10], close_ids=[11], pad_id=0, _encode_many=encode_many)
    encoder = OmniJevEmbeddingEncoder(mso, basis, embed_dim=8, train_gate=True)
    encoder.train()
    assert not backbone.training
    result = encoder({'images': ['unused-toy-image']})
    assert result.slots.shape == (2, 8)
    assert backbone.calls == 2  # shared prefix, then one batched suffix call
    (result.embedding * torch.randn(16)).sum().backward()
    assert backbone.embedding.weight.grad is None
    assert encoder.embedding_head.opt[0].weight.grad.abs().sum() > 0
    assert encoder.embedding_head.projection[1].weight.grad.abs().sum() > 0
    torch.testing.assert_close(encoder.encode({'images': ['unused-toy-image']}).slots, result.slots)


def test_lora_path_reaches_only_enabled_adapter_parameters():
    class AdaptedToy(ToyBackbone):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Linear(12, 4, bias=False)
            self.lora_B = nn.Linear(4, 12, bias=False)

        def forward(self, input_ids, past_key_values=None, **kwargs):
            x = self.embedding(input_ids)
            h = (x + self.lora_B(self.lora_A(x))).cumsum(dim=1)
            if past_key_values is not None:
                h = h + past_key_values.layers[0].state[:, None]
            return SimpleNamespace(
                past_key_values=SimpleNamespace(layers=[SimpleNamespace(state=h[:, -1])]),
                hidden_states=[h],
            )

    backbone = AdaptedToy()
    model = SimpleNamespace(backbone=backbone, head=OptionScorer(12, 16), rope_positions=lambda *args: None)
    mso = SimpleNamespace(branch=True, model=model, dev=torch.device('cpu'), open_ids=[10], close_ids=[11], pad_id=0,
                          _encode_many=lambda *args: ([["color"]], [[{}]], [[1, 2, 3, 10, 4, 11]], [([3], [5])], {}))
    encoder = OmniJevEmbeddingEncoder(mso, AttributeBasis(('color',)), embed_dim=8, tune_lora=True)
    encoder.train()
    assert backbone.training
    result = encoder({'images': ['unused-toy-image']})
    (result.embedding * torch.randn(8)).sum().backward()
    assert backbone.lora_A.weight.grad.abs().sum() > 0
    assert backbone.lora_B.weight.grad.abs().sum() > 0
    assert backbone.embedding.weight.grad is None
    assert encoder.embedding_head.projection[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('directions', [(), ('color', 'color'), ('x'*201,), ('<|opt|>',)])
def test_invalid_basis_rejected(directions):
    with pytest.raises(ValueError):
        AttributeBasis(directions)
