import torch

from scripts.train_cached import retrieval_metrics


def test_retrieval_excludes_self_and_reports_singleton_coverage():
    # Pairs share a direction; the fifth item has no positive in this gallery.
    slots = torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 1.]], [[0., 1.]], [[-1., 0.]]])
    labels = torch.tensor([[0], [0], [1], [1], [2]])
    result = retrieval_metrics(slots, labels, ['color'])
    assert result['color']['eligible_queries'] == 4
    assert result['color']['total_queries'] == 5
    assert result['macro'] == {'R@1': 1.0, 'R@5': 1.0, 'mAP': 1.0}


def test_retrieval_ranking_measures_attribute_mismatch():
    # Geometric nearest neighbours have different labels for every query.
    slots = torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 1.]], [[0., 1.]]])
    labels = torch.tensor([[0], [1], [0], [1]])
    result = retrieval_metrics(slots, labels, ['color'])
    assert result['color']['R@1'] == 0.0
    assert result['color']['R@5'] == 1.0
    assert result['color']['mAP'] < 1.0


def test_joint_relevance_requires_all_attributes_and_training_pools_stay_in_split():
    from scripts.experiment_suite import evaluate,fast_pools
    labels=torch.tensor([[0,0,0],[0,0,0],[0,0,1],[0,1,1],[1,1,1],[1,1,1]])
    slots=torch.nn.functional.normalize(torch.randn(6,3,5),dim=-1)
    vectors=torch.nn.functional.normalize(slots.flatten(1),dim=-1)
    result=evaluate(slots,vectors,labels,['a','b','c'])
    assert result['joint']['eligible_queries']==4  # rows 2/3 share some, not all, labels
    assert result['attributes']['a']['eligible_queries']==6
    train=[0,1,4,5]
    pools=fast_pools(labels.tolist(),train,joint=True)
    for a,pos,neg in pools[0]:
        assert {a,*pos,*neg}.issubset(train)
        assert all(torch.equal(labels[a],labels[p]) for p in pos)
        assert all(not torch.equal(labels[a],labels[n]) for n in neg)
