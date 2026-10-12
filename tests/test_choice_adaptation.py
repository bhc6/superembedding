import json
from pathlib import Path
import pytest
from scripts.evaluate_choice_adaptation import class_metrics,paired_stats
from superembedding.general import GENERAL_BASIS
from superembedding.heads import AttributeBasis


def test_sealed_choices_preserve_slots_and_original_instruction():
    design=json.loads(Path('sources/choice_adaptation_v5.json').read_text(encoding='utf-8'))
    assert design['directions']['original']==list(GENERAL_BASIS.directions)
    assert design['instruction']==GENERAL_BASIS.instruction
    for arm in design['arms']:
        for source in design['sources']:
            d=design['directions'][arm]
            d=d[source] if isinstance(d,dict) else d
            assert len(AttributeBasis(tuple(d),design['instruction']).directions)==4


def test_majority_accuracy_is_not_balanced_accuracy():
    rows=[{'id':str(i),'key':'yes' if i<8 else 'no'} for i in range(10)]
    predictions=[{'id':r['id'],'candidate_id':'0','correct':r['key']=='yes'} for r in rows]
    result=class_metrics(rows,{'predictions':predictions})
    assert result['balanced_accuracy']==.5
    assert result['predicted_class_counts']=={'yes':10}
    assert result['per_class_recall']=={'yes':1.,'no':0.}


def test_paired_differences_keep_signed_improvements():
    result=paired_stats([1,0,-1,0])
    assert result['delta']==0
    assert result['paired_bootstrap_95'][0]<0<result['paired_bootstrap_95'][1]
    assert paired_stats([1]*5)['delta']==pytest.approx(1)
