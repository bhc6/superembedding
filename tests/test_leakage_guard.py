import hashlib
import json
import importlib
import pytest
from scripts.audit_data_leakage import distinct_split_pairs, normalize_text, informative
from scripts.prepare_leakage_quarantine import quarantine
from scripts.audit_question_overlap import train_lookup, most_common
from superembedding.leakage_guard import promotion_problems


def test_cross_split_includes_val_test_but_not_same_split():
    rows=[{'split':s} for s in ['train','train','val','test']]
    assert distinct_split_pairs(rows,{'same':[0,1,2,3]}) == {(0,2),(0,3),(1,2),(1,3),(2,3)}
    assert not informative(normalize_text(' YES. '))


def test_quarantine_transitive_keeps_test_and_original_rows():
    rows=[{'id':k,'split':s} for k,s in [('a','train'),('b','val'),('c','test'),('d','train')]]
    kept,groups=quarantine(rows,[{'a':'a','b':'b'},{'a':'b','b':'c'}])
    assert [r['id'] for r in kept]==['c','d']
    assert groups[0]['quarantined_ids']==['a','b']
    assert len(rows)==4


def test_lookup_never_fits_held_answers():
    rows=[{'split':s,'task':'qa','source':'x','texts':[{'user':'Where?', 'assistant':a}]} for s,a in [('train','Left'),('val','Right'),('test','Right')]]
    questions,counts=train_lookup(rows)
    assert most_common(questions[('x','where')])=='left'
    assert counts['x']=={'left':1}


def test_promotion_missing_stale_unreviewed_and_blocked(tmp_path):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=tmp_path/'records.jsonl';manifest.write_text('sample')
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps({'provenance':{'manifests':{'test':{'path':str(manifest),'sha256':sha(manifest)}}}}))
    ledger=tmp_path/'review.json'
    assert promotion_problems(ledger,root=tmp_path)
    review={'schema_version':1,'audit':{'path':str(audit),'sha256':sha(audit)},
            'datasets':{'test':{'status':'cleared','unresolved':[]}},'release_eligible':True,
            'blockers':[],'confirmatory_test_status':'sealed_unseen'}
    ledger.write_text(json.dumps(review))
    assert not promotion_problems(ledger,manifest,root=tmp_path)
    assert promotion_problems(ledger,tmp_path/'unknown',root=tmp_path)
    review['datasets']['test']['status']='blocked';ledger.write_text(json.dumps(review))
    assert promotion_problems(ledger,root=tmp_path)
    review['datasets']['test']['status']='cleared';ledger.write_text(json.dumps(review))
    manifest.write_text('changed')
    assert any('manifest changed' in p for p in promotion_problems(ledger,root=tmp_path))
    audit.write_text('{}')
    assert any('screen changed' in p for p in promotion_problems(ledger,root=tmp_path))


@pytest.mark.parametrize('module,argv',[
    ('scripts.publish_github',['publish_github.py']),
    ('scripts.finish_general_stage',['finish_general_stage.py','--run','unused']),
])
def test_promotion_entrypoints_stop_before_network_or_export(monkeypatch,module,argv):
    entry=importlib.import_module(module)
    def blocked(*args,**kwargs):
        raise RuntimeError('review-blocked')
    def forbidden(*args,**kwargs):
        pytest.fail('Subprocess or credential lookup ran before leakage clearance')
    monkeypatch.setattr('superembedding.leakage_guard.require_promotion_clearance',blocked)
    monkeypatch.setattr(entry.sys,'argv',argv)
    monkeypatch.setattr(entry.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='review-blocked'):
        entry.main()
