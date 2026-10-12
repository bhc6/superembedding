import numpy as np
from scripts.report_evidence_comparison import paired_summary


def test_identical_models_have_exact_zero_paired_uncertainty():
    samples=np.random.default_rng(1).integers(0,4,size=(100,4))
    result=paired_summary([0,.25,.75,1],[0,.25,.75,1],samples)
    assert result['paired_cluster_bootstrap_95']==[0.,0.]
    assert result['minus_reference']==0.


def test_constant_scene_difference_is_preserved_in_all_bootstraps():
    samples=np.random.default_rng(1).integers(0,4,size=(100,4))
    result=paired_summary([.25,.5,.75,1],[0,.25,.5,.75],samples)
    assert result['paired_cluster_bootstrap_95']==[.25,.25]
    assert result['rate']==.625
