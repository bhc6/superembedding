import torch
from superembedding.general_ablation import ARCHITECTURES,GeneralAblationHead


def test_equal_projection_starts_at_identical_similarity():
    torch.manual_seed(17);a=GeneralAblationHead('shared_concat',width=16,slot_dim=8)
    torch.manual_seed(17);b=GeneralAblationHead('output_equal',width=16,slot_dim=8)
    x=torch.randn(6,4,16)
    assert torch.equal(a(x),b(x))


def test_independent_equal_starts_shared_but_can_diverge():
    torch.manual_seed(17);a=GeneralAblationHead('shared_concat',width=16,slot_dim=8)
    torch.manual_seed(17);b=GeneralAblationHead('independent_equal',width=16,slot_dim=8)
    x=torch.randn(6,4,16)
    assert torch.allclose(a(x),b(x),atol=1e-7)
    b(x)[0,0].backward()
    assert b.independent[0][1].weight.grad is not None
    assert b.independent[0][1].weight.data_ptr()!=b.independent[1][1].weight.data_ptr()


def test_all_heads_produce_finite_normalized_vectors():
    for name in ARCHITECTURES:
        head=GeneralAblationHead(name,width=16,slot_dim=8)
        v=head(torch.randn(5,4,16))
        assert v.shape==(5,8 if name in {'output_small','independent_matched'} else 32)
        assert torch.allclose(v.norm(dim=-1),torch.ones(5),atol=1e-6)


def test_mean_pool_is_invariant_to_branch_permutation():
    head=GeneralAblationHead('mean_pool',width=16,slot_dim=8);x=torch.randn(5,4,16)
    assert torch.allclose(head(x),head(x[:,[3,1,0,2]]),atol=1e-6)
