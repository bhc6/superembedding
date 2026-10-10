import random
import torch
import pytest
from superembedding.research import relation,rank_metrics,evaluate,prepare_relations,ObservedTriplets
from superembedding.research import random_ranking_expectation

def labels():
    return [[{'red','brown'},{'long'}],[{'red'},{'long'}],[{'blue'},{'short'}],
            [set(),{'long'}],[{'blue'},{'long'}]]

def test_missing_labels_and_set_overlap():
    positive,valid=relation(labels(),(0,))
    assert positive[0,1] and not positive[0,2]
    assert not valid[3].any() and not valid[:,3].any()
    assert not valid.diag().any()
    joint,_=relation(labels(),(0,1))
    assert joint[0,1] and not joint[2,4]

def test_ranking_excludes_unknown_and_self():
    lab=[[{'a'}],[{'a'}],[{'b'}],[set()]]
    positive,valid=relation(lab,(0,))
    vectors=torch.nn.functional.normalize(torch.tensor([[1.,0.],[.8,.2],[0.,1.],[1.,0.]]),dim=-1)
    result=rank_metrics(vectors,positive,valid,chunk=2)
    assert result['mAP']==1 and result['R@1']==1
    assert result['eligible_queries']==2 and result['observed_queries']==3
    summary=evaluate(vectors[:,None],vectors,['one'],prepare_relations(lab,['one']))
    assert summary['pairs']=={} and summary['attributes']['macro']['mAP']==1

def test_analytic_chance_map_matches_enumeration():
    positive=torch.tensor([[False,True,True,False]])
    valid=torch.tensor([[False,True,True,True]])
    result=random_ranking_expectation(positive,valid)
    assert abs(result['mAP']-(1+(1+2/3)/2+(.5+2/3)/2)/3)<1e-12
    assert abs(result['R@1']-2/3)<1e-12 and result['R@5']==1

@pytest.mark.parametrize('joint',[False,True])
def test_sampler_only_uses_observed_train_pairs(joint):
    lab=labels()+[[{'red'},{'long'}]]
    pool=ObservedTriplets(lab,range(5),joint)
    ids,attrs=pool.sample(1000,random.Random(42))
    assert 5 not in ids
    for (a,p,n),k in zip(ids.tolist(),attrs.tolist()):
        dims=range(2) if joint else [k]
        assert a!=p and all(lab[a][d]&lab[p][d] for d in dims)
        assert all(lab[n][d] and lab[a][d] for d in dims)
        assert not all(lab[a][d]&lab[n][d] for d in dims)
