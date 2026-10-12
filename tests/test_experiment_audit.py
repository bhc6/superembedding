import pytest
import torch
from scripts.audit_experiment_completeness import check_predictions
from superembedding.validation_protocol import score_vectors


def fixture():
    items=[{'id':str(i),'source':'qa','key':str(i),'type':'other'} for i in range(3)]
    return items,score_vectors(items,torch.eye(3),torch.eye(3))


def test_audit_rejects_missing_queries_even_if_macro_is_unchanged():
    items,result=fixture();result['predictions'].pop()
    with pytest.raises(ValueError,match='Missing'):check_predictions(items,result)


def test_audit_rejects_candidate_or_relevance_corruption():
    items,result=fixture();result['predictions'][0]['candidate_id']='1'
    with pytest.raises(ValueError,match='relevance'):check_predictions(items,result)


def test_complete_prediction_and_subgroup_coverage():
    items,result=fixture();check_predictions(items,result)
    result['per_source']['qa']['per_question_type']['other']['queries']=2
    with pytest.raises(ValueError,match='Question-type'):check_predictions(items,result)
