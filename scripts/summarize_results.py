"""Build reviewable tables and copy compact evidence into version control."""
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'artifacts/results';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for family in ['omnijev','base','flair']:
        path=ROOT/f'runs/fashion2400/results_{family}.json'
        report=json.loads(path.read_text())
        shutil.copy2(path,out/path.name)
        shutil.copy2(path.with_suffix('.selection.json'),out/path.with_suffix('.selection.json').name)
        frozen=report['frozen_features']
        rows.append([family+' frozen',f"{frozen['attributes']['macro']['mAP']*100:.2f}",
                     f"{statistics.mean(p['mAP'] for p in frozen['pairs'].values())*100:.2f}",
                     f"{frozen['joint']['mAP']*100:.2f}",'—','—'])
        if report.get('global_features'):
            g=report['global_features']
            rows.append([family+' global frozen',f"{g['attributes']['macro']['mAP']*100:.2f}",
                f"{statistics.mean(p['mAP'] for p in g['pairs'].values())*100:.2f}",f"{g['joint']['mAP']*100:.2f}",'—','512'])
        for arch,r in report['architectures'].items():
            s=r['summary']
            fmt=lambda key:f"{s[key]['mean']*100:.2f} ± {s[key]['std']*100:.2f}"
            rows.append([family+'/'+arch,fmt('attribute_mAP'),fmt('pair_mAP'),fmt('joint_mAP'),
                         str(r['lr']),str(r['seeds'][0]['global_dim'])])
    lora=ROOT/'runs/fashion2400/lora_raw/test_results.json'
    if lora.exists():
        r=json.loads(lora.read_text()); m=r['test']
        training=json.loads((lora.parent/'training.json').read_text())
        label=f"LoRA validation-selected step {training['best_step']} (1 seed)"
        rows.append([label,f"{m['attributes']['macro']['mAP']*100:.2f}",
            f"{statistics.mean(p['mAP'] for p in m['pairs'].values())*100:.2f}",f"{m['joint']['mAP']*100:.2f}",'1e-5','768'])
        shutil.copy2(lora,out/'lora_test_results.json')
        shutil.copy2(lora.parent/'training.json',out/'lora_training.json')
        if (lora.parent/'export_manifest.json').exists():
            shutil.copy2(lora.parent/'export_manifest.json',out/'lora_export_manifest.json')
    lines=['# Formal local comparison — Fashion2400','',
           'mAP in percent; ± is sample standard deviation across training seeds 17/42/91. '
           'Same 469 held-out queries/gallery images; self excluded. These are local image-to-image '
           'attribute retrieval measurements, not the original papers’ benchmark results.','',
           '| Model | Attribute mAP | Two-attribute mAP | Three-attribute mAP | Selected LR | Global dim |',
           '|---|---:|---:|---:|---:|---:|']
    lines += ['| '+' | '.join(row)+' |' for row in rows]
    (ROOT/'reports/FORMAL_RESULTS_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifests=ROOT/'artifacts/data';manifests.mkdir(parents=True,exist_ok=True)
    for name in ['records.jsonl','metadata.json']:
        shutil.copy2(ROOT/'data/fashion2400'/name,manifests/name)
    validations=out/'validation';validations.mkdir(exist_ok=True)
    for family,suite in [('omnijev','suite'),('base','suite_base'),('flair','suite_flair')]:
        for path in (ROOT/f'runs/fashion2400/{suite}').glob('*/validation.json'):
            shutil.copy2(path,validations/f'{family}_{path.parent.name}.json')
    for name in ['resume_validation.json','cuda_check.json']:
        shutil.copy2(ROOT/'runs'/name,out/name)
    shutil.copy2(ROOT/'runs/fashion2400/ledger.jsonl',out/'ledger.jsonl')
    pilot=ROOT/'artifacts/development';pilot.mkdir(exist_ok=True)
    for name in ['raw','gated','gated_tuned']:
        shutil.copy2(ROOT/f'runs/fashion600/{name}/metrics.json',pilot/f'{name}_metrics.json')
    shutil.copy2(ROOT/'runs/fashion600/lora_smoke/training.json',pilot/'lora_smoke_training.json')
    shutil.copy2(ROOT/'runs/fashion600/reload_validation.json',pilot/'reload_validation.json')
    print('\n'.join(lines))


if __name__=='__main__':main()
