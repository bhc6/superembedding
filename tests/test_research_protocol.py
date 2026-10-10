"""Protocol invariants that prevent silent budget and output-size mismatches."""
from scripts.audit_research import architectures
from scripts.research_suite import create
from superembedding.research import relation,rank_metrics
import torch

def test_grid_has_432_runs():
    assert 3*3*2*sum(len(architectures(b,s)) for b in ('omnijev','base','flair') for s in (25,50,100))==432

def test_same_seed_projection_comparators_share_slot_initialization():
    for k in (1,3):
        heads=[create(arch,32,k,None,17) for arch in ('joint_concat','joint_out_equal','joint_out_small')]
        for h in heads[1:]:
            for name,p in heads[0].projection.state_dict().items():
                assert torch.equal(p,h.projection.state_dict()[name])
        u=torch.randn(5,k,32)
        assert heads[0](u).embedding.shape==(5,k*256)
        assert heads[1](u).embedding.shape==(5,k*256)
        assert heads[2](u).embedding.shape==(5,128 if k==1 else 256)
        assert torch.equal(heads[0](u).slots,heads[1](u).slots)

def test_full_map_known_nonperfect_ranking():
    # query 0 positives at ranks 1 and 3 -> AP=(1+2/3)/2.
    vectors=torch.tensor([[1.,0.],[.9,.1],[.8,.2],[.7,.3]])
    vectors=torch.nn.functional.normalize(vectors,dim=-1)
    positive=torch.zeros(4,4,dtype=torch.bool); positive[0,1]=positive[0,3]=True
    valid=~torch.eye(4,dtype=torch.bool)
    result=rank_metrics(vectors,positive,valid)
    assert abs(result['mAP']-5/6)<1e-6
    assert result['eligible_queries']==1
