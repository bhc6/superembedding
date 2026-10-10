"""Package all trained heads and resumable last checkpoints; never include image data."""
import hashlib
import json
from pathlib import Path
import zipfile
import argparse

ROOT=Path(__file__).resolve().parents[1]


def archive(name, paths):
    output=ROOT/'release_assets'/name
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for path in sorted(set(paths)):
            if path.is_file(): z.write(path,path.relative_to(ROOT).as_posix())
        for path in [ROOT/'LICENSE',ROOT/'NOTICE',ROOT/'MODEL_CARD.md',ROOT/'models/OmniJev-0.8B-v1.1/LICENSE']:
            z.write(path,path.relative_to(ROOT).as_posix())
    if output.stat().st_size>=2_000_000_000: raise RuntimeError('Split archive: GitHub asset size limit')
    digest=hashlib.file_digest(output.open('rb'),'sha256').hexdigest()
    return {'file':name,'bytes':output.stat().st_size,'sha256':digest}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--heads-only',action='store_true')
    args=ap.parse_args()
    (ROOT/'release_assets').mkdir(exist_ok=True)
    suites=[ROOT/'runs/fashion2400'/s for s in ['suite','suite_base','suite_flair']]
    folders=[p.parent for s in suites for p in s.glob('*/validation.json')]
    if len(folders)!=54: raise ValueError(f'Expected all 54 cached-feature experiments, got {len(folders)}')
    evidence=[]
    for folder in folders:
        for filename in ['best_head.pt','last.pt','validation.json']:
            p=folder/filename
            if not p.exists(): raise FileNotFoundError(p)
            evidence.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
                             'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
    (ROOT/'artifacts/results/checkpoint_manifest.json').write_text(json.dumps(evidence,indent=2))
    assets=[]
    assets.append(archive('all_trained_heads.zip',[p/f for p in folders for f in ['best_head.pt','validation.json']]))
    for family,suite in zip(['omnijev','base','flair'],suites):
        assets.append(archive(f'{family}_resume_checkpoints.zip',list(suite.glob('*/last.pt'))))
    lora=ROOT/'runs/fashion2400/lora_raw'
    if args.heads_only:
        (ROOT/'release_assets/heads_manifest.json').write_text(json.dumps(assets,indent=2))
        print(json.dumps(assets,indent=2));return
    if not (lora/'test_results.json').exists(): raise ValueError('LoRA evaluation incomplete')
    files=list((lora/'adapter').rglob('*'))+list((lora/'final_adapter').rglob('*'))+[lora/f for f in ['embedding_head.pt','final_embedding_head.pt','export_manifest.json','training.json','test_results.json','best.pt','last.pt','reload_reference.pt','final_reload_reference.pt']]
    assets.append(archive('lora_model_and_resume.zip',files))
    dev=ROOT/'runs/fashion600'
    development=[p for name in ['raw','gated','gated_tuned'] for p in (dev/name).glob('*') if p.suffix in {'.pt','.json'}]
    development+=list((dev/'lora_smoke/adapter').rglob('*'))
    development += [dev/'lora_smoke'/f for f in ['embedding_head.pt','training.json','reload_reference.pt']]
    development += [dev/'reload_validation.json']
    assets.append(archive('development_models_fashion600.zip',development))
    manifest=ROOT/'release_assets/SHA256SUMS.json'
    manifest.write_text(json.dumps(assets,indent=2))
    print(json.dumps(assets,indent=2))


if __name__=='__main__':main()
