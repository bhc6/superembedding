"""Archive every v3 trained head and recovery state, without dataset images."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='release_assets/v3'); args=ap.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    seal=json.loads((ROOT/'artifacts/results/v3/selection_sealed.json').read_text())
    results=json.loads((ROOT/'artifacts/results/v3/test_results.json').read_text())
    assert len(results['results'])==216 and results['selection_fingerprint']==seal['selection_fingerprint']
    exports=json.loads((ROOT/'artifacts/results/v3/export_verification.json').read_text())
    inference=json.loads((ROOT/'artifacts/results/v3/fresh_inference_verification.json').read_text())
    assert exports['count']==432 and exports['maximum_metric_difference']<=1e-6
    assert len(inference['checks'])==12
    folders=sorted(p.parent for p in (ROOT/'runs/v3').glob('*/suite/*/*/*/validation.json'))
    assert len(folders)==432
    evidence=[]; assets=[]
    def archive(name,files):
        path=out/name
        members={f.relative_to(ROOT).as_posix():f for f in files}
        for filename in ('LICENSE','NOTICE','MODEL_CARD.md','reports/MODEL_CARD_V3.md','reports/REPRODUCING_V3.md',
                         'reports/EXPERIMENT_PROTOCOL_V3.zh.md','reports/PROJECTION_DESIGN_V3.zh.md',
                         'reports/RESULTS_V3.md','reports/CONCLUSIONS_V3.zh.md'):
            members[filename]=ROOT/filename
        members['third_party/OmniJev-LICENSE.txt']=ROOT/'models/OmniJev-0.8B-v1.1/LICENSE'
        if not path.exists():
            temp=path.with_suffix('.tmp')
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                for member,f in sorted(members.items()): z.write(f,member)
            temp.replace(path)
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None: raise ValueError('Invalid release archive')
            if set(z.namelist())!=set(members): raise ValueError('Archive member list changed')
            for member,f in members.items():
                actual=hashlib.sha256(z.read(member)).hexdigest()
                expected=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
                if actual!=expected: raise ValueError('Stale/damaged archive member')
        if path.stat().st_size>=2_000_000_000: raise RuntimeError('Asset exceeds GitHub size limit')
        assets.append({'file':name,'bytes':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
        print(json.dumps(assets[-1]),flush=True)
    for folder in folders:
        for name in ('best_head.pt','last.pt','validation.json'):
            path=folder/name; evidence.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
                'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
    for dataset in ('fashion_expanded','cub_attributes','dtd'):
        subset=[p for p in folders if p.relative_to(ROOT/'runs/v3').parts[0]==dataset]
        archive(f'{dataset}_all_heads.zip',[p/name for p in subset for name in ('best_head.pt','validation.json')])
        # Split optimizer states by backend to keep each GitHub asset comfortably <2GB.
        for backend in ('omnijev','base','flair'):
            archive(f'{dataset}_{backend}_resume.zip',[p/'last.pt' for p in subset if p.relative_to(ROOT/'runs/v3').parts[2]==backend])
    (ROOT/'artifacts/results/v3/checkpoint_manifest.json').write_text(json.dumps(evidence,indent=2))
    (out/'SHA256SUMS.json').write_text(json.dumps(assets,indent=2))

if __name__=='__main__': main()
