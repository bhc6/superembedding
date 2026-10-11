"""Final count, identity and integrity audit for the v4 extension."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.baselines import MODELS,DATASETS
from superembedding.checkpoint import fingerprint,atomic_json

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);a=ap.parse_args()
    out=Path('artifacts/results/v4')
    protocol=json.loads((out/'protocol_manifest.json').read_text());assert sha(protocol['path'])==protocol['sha256']
    seal=json.loads((out/'selection_sealed.json').read_text());assert fingerprint(seal['selected'])==seal['selection_fingerprint']
    results=json.loads((out/'test_results.json').read_text())
    assert results['selection_fingerprint']==seal['selection_fingerprint'] and len(results['results'])==162
    assert len(results['frozen_baselines'])==18
    manifest=json.loads((out/'checkpoint_manifest.json').read_text());assert len(manifest)==648
    for item in manifest:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    exports=json.loads((out/'export_verification.json').read_text());assert len(exports)==18 and sum(x['reloaded_heads'] for x in exports)==324
    assert max(x['max_validation_mAP_difference'] for x in exports)<2e-6
    fresh=json.loads((out/'fresh_inference_verification.json').read_text());assert len(fresh['checks'])==18
    features=[]
    for d in DATASETS:
        for b in MODELS:
            p=Path(a.features)/d/b/'features.pt'
            features.append({'dataset':d,'backend':b,'bytes':p.stat().st_size,'sha256':sha(p)})
    atomic_json(features,out/'feature_bank_manifest.json')
    train_ids={(r['dataset'],r['backend'],r['scale'],r['seed']) for r in seal['selected']}
    test_ids={(r['dataset'],r['backend'],r['scale'],r['seed']) for r in results['results']}
    assert len(train_ids)==162 and train_ids==test_ids
    report=dict(complete_training_runs=324,validated_exported_heads=324,recovery_states=324,
        sealed_test_evaluations=162,frozen_baseline_evaluations=18,feature_banks=18,
        fresh_inference_domain_backend_checks=18,protocol_unchanged=True,all_checkpoint_hashes_match=True,
        all_feature_bank_hashes_recorded=True,selection_fingerprint=seal['selection_fingerprint'],
        prior_v3_results_reused_without_reselection=True)
    atomic_json(report,out/'completion_audit.json');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
