"""Record image-reviewed decisions without changing historical data or scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json
from superembedding.leakage_guard import digest, ROOT
from scripts.audit_data_leakage import collect


def main():
    path = ROOT/'artifacts/results/v5/data_leakage_audit.json'
    audit = json.loads(path.read_text(encoding='utf-8'))
    families, manifests = collect()
    if manifests != audit['provenance']['manifests']:
        raise ValueError('Data changed after screening')
    decisions = []

    def decision(family, a, b, verdict, reason):
        lookup = {r['id']: r for r in families[family]}
        pairs = audit['families'][family]['perceptual_candidates']
        if not any({p['a'], p['b']} == {a, b} for p in pairs):
            raise ValueError('Reviewed pair absent from screen: ' + a + ', ' + b)
        decisions.append({'family': family, 'a': a, 'b': b,
                          'a_split': lookup[a]['split'], 'b_split': lookup[b]['split'],
                          'a_file_sha256': digest(lookup[a]['image']),
                          'b_file_sha256': digest(lookup[b]['image']),
                          'verdict': verdict, 'reason': reason,
                          'reviewer': 'assistant visual inspection; not independent human adjudication'})

    decision('cub_attributes', 'cub_01413', 'cub_01460', 'same_photo', 'Same black bird, branch and background; encoding differs.')
    decision('cub_attributes', 'cub_10371', 'cub_10379', 'same_photo', 'Same yellow bird, foliage and composition; encoding differs.')
    decision('cub_attributes', 'cub_11592', 'cub_11594', 'same_scene', 'Same bird/branch/background, differing beak pose; burst/scene dependence.')
    for a, b in [('potholed_potholed_0144','potholed_potholed_0045'),
                 ('dotted_dotted_0131','dotted_dotted_0060'),
                 ('grid_grid_0035','meshed_meshed_0044'),
                 ('wrinkled_wrinkled_0037','wrinkled_wrinkled_0023'),
                 ('lined_lined_0047','lined_lined_0023'),
                 ('knitted_knitted_0102','knitted_knitted_0091'),
                 ('crosshatched_crosshatched_0092','crosshatched_crosshatched_0093'),
                 ('pitted_pitted_0002','pitted_pitted_0124')]:
        decision('dtd', 'dtd_'+a, 'dtd_'+b, 'same_photo', 'Visually matching texture detail and composition; resized/recompressed versions.')
    for a,b in [('chart2text:20016','chartqa:12515'), ('chart2text:19695','chart2text:1329'),
                ('chartqa:17976','chartqa:3692')]:
        decision('v5', a, b, 'different_content_shared_template', 'Different chart topics/labels/values; do not delete by low hash distance alone.')
    for a,b in [('scienceqa_official:681','scienceqa:4772'), ('scienceqa_official:152','scienceqa:1499'),
                ('scienceqa_official:322','scienceqa:1499')]:
        decision('v5', a, b, 'different_content_shared_template', 'Same US map canvas, different city marker locations; reviewed pair only.')
    decision('v5', 'docvqa:7503', 'docvqa:7512', 'same_entity_fixed_answer',
             'Different dates/products/amounts, same BIOMET company/vendor and PO box question/answer. Not identical documents; entity generalization is confounded.')
    statuses = {}
    for name in families:
        statuses[name] = {'status': 'blocked', 'unresolved': [],
                          'historical_result_use': 'exploratory_only_not_leakage_cleared'}
    for name in ['fashion600', 'fashion2400', 'fashion_expanded']:
        statuses[name]['unresolved'] = ['Product identity/variant and crop-aware review; full-image hashes are insufficient.']
    for name in ['cub_attributes', 'dtd']:
        statuses[name]['unresolved'] = ['Confirmed cross-split photo/scene dependence. New versioned split, fresh head training and all matched baseline reruns required.',
                                        'Conservative removal of screened candidates does not exhaust crop/scene duplicates.']
    statuses['v5']['unresolved'] = ['DocVQA entity/answer grouping missing; current validation selection is exploratory.',
                                   'Remaining ScienceQA shared-map candidates require content review; OOD and external are related sources.',
                                   'Question-only overlap and language-shortcut controls are diagnostics, not independent entity holdout.',
                                   'Current external results already observed; new method choices need a separately sealed unseen test.']
    review = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(),
              'audit': {'path': str(path.relative_to(ROOT)).replace('\\','/'), 'sha256': digest(path)},
              'datasets': statuses, 'pair_decisions': decisions,
              'entity_group': {'family': 'v5', 'members': ['docvqa:7461','docvqa:7499','docvqa:7503','docvqa:7512'],
                               'scope': 'Verified examples, not an exhaustive inventory of this entity'},
              'blockers': ['Cross-split duplicates in CUB/DTD', 'V5 DocVQA entity overlap',
                           'Remaining product/document/scene review', 'Seen tests cannot validate subsequent design choices'],
              'release_eligible': False, 'confirmatory_test_status': 'not_sealed',
              'prediction_or_metric_used_for_filtering': False,
              'historical_files_modified': False,
              'pretraining_contamination': 'Unknown; no local audit can clear inherited/proprietary training corpora.'}
    atomic_json(review, ROOT/'artifacts/results/data_leakage_review.json')
    print(json.dumps({'reviewed_pairs':len(decisions),'release_eligible':False}))


if __name__ == '__main__':
    main()
