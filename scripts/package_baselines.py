"""Archive every v4 trained head and complete recovery state, without secrets/images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.baselines import MODELS
from superembedding.checkpoint import atomic_json

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    result=ROOT/'artifacts/results/v4'
    seal=json.loads((result/'selection_sealed.json').read_text());tests=json.loads((result/'test_results.json').read_text())
    assert len(tests['results'])==162 and tests['selection_fingerprint']==seal['selection_fingerprint']
    exports=json.loads((result/'export_verification.json').read_text());assert sum(x['reloaded_heads'] for x in exports)==324
    assert max(x['max_validation_mAP_difference'] for x in exports)<2e-6
    assert len(json.loads((result/'fresh_inference_verification.json').read_text())['checks'])==18
    for b in ('clip_b32','qwen3_vl_instruct'):
        resume=json.loads((result/f'resume_{b}.json').read_text());assert resume['max_model_abs_difference']==0
        assert all(resume[k] for k in ('model','optimizer','rng_local','rng_python','rng_torch','rng_cuda','extra'))
    folders=sorted((ROOT/'runs/v4').glob('*/suite/*/*/*/validation.json'));assert len(folders)==324
    common=[ROOT/'LICENSE',ROOT/'NOTICE',ROOT/'reports/MODEL_CARD_V4.md',ROOT/'reports/REPRODUCING_V4.md',
        ROOT/'reports/EXPERIMENT_PROTOCOL_V4.zh.md',ROOT/'reports/RESULTS_V4.md',ROOT/'reports/CONCLUSIONS_V4.zh.md',
        ROOT/'sources/baseline_models_v4.json',ROOT/'sources/baseline_architectures_v4.json']
    assets=[]
    for backend in MODELS:
        reports=[p for p in folders if p.parent.parent.parent.name==backend];assert len(reports)==54
        files=common+[p.parent/name for p in reports for name in ('best_head.pt','last.pt','validation.json')]
        members={p.relative_to(ROOT).as_posix():p for p in files}
        target=out/f'{backend}_all_heads_and_resume.zip'
        if not target.exists():
            temp=target.with_suffix('.tmp')
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                for name,p in sorted(members.items()):z.write(p,name)
            temp.replace(target)
        with zipfile.ZipFile(target) as z:
            assert set(z.namelist())==set(members)
            for name,p in members.items():
                assert hashlib.sha256(z.read(name)).hexdigest()==sha(p),name
        assert target.stat().st_size<2_000_000_000
        asset={'file':target.name,'sha256':sha(target),'bytes':target.stat().st_size}
        assets.append(asset);print(json.dumps(asset),flush=True)
    atomic_json(assets,out/'SHA256SUMS.json');atomic_json(assets,result/'release_assets.json')

if __name__=='__main__':main()
