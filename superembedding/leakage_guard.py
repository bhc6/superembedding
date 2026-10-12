"""Fail-closed eligibility check; numerical reproducibility is a separate audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'artifacts/results/data_leakage_review.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def promotion_problems(ledger=LEDGER, manifest=None, root=ROOT):
    root = Path(root)
    problems = []
    try:
        review = json.loads(Path(ledger).read_text(encoding='utf-8'))
        if review.get('schema_version') != 1:
            problems.append('Unknown review schema')
        audit_path = root / review['audit']['path']
        if digest(audit_path) != review['audit']['sha256']:
            problems.append('Leakage screen changed after review')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        audited = audit['provenance']['manifests']
        covered = {name for name in review['datasets']}
        if covered != set(audited):
            problems.append('Review does not cover every screened dataset')
        matched = manifest is None
        for name, entry in audited.items():
            path = root / entry['path']
            if digest(path) != entry['sha256']:
                problems.append('Audited manifest changed: ' + name)
            if manifest is not None and path.resolve() == Path(manifest).resolve():
                matched = True
            state = review['datasets'].get(name, {})
            if state.get('status') != 'cleared' or state.get('unresolved'):
                problems.append('Dataset not cleared: ' + name)
        if not matched:
            problems.append('Requested manifest was not reviewed')
        if review.get('release_eligible') is not True or review.get('blockers'):
            problems.append('Research release is blocked by leakage/selection review')
        if review.get('confirmatory_test_status') != 'sealed_unseen':
            problems.append('No sealed unseen confirmatory test')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append('Missing or invalid leakage review: ' + str(exc))
    return problems


def require_promotion_clearance(manifest=None):
    problems = promotion_problems(manifest=manifest)
    if problems:
        raise RuntimeError('Formal evaluation/promotion blocked: ' + '; '.join(problems))
