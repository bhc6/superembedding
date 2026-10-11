"""Produce v4 comparisons from sealed measurements, retaining v3 unchanged."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.baselines import DATASETS,MODELS
from superembedding.checkpoint import atomic_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'fashion_expanded':'Fashion','cub_attributes':'CUB','dtd':'DTD'}
LABELS={'omnijev':'OmniJev raw','base':'Qwen3.5 base raw','flair':'FLAIR',
    'clip_b32':'CLIP B/32','siglip2_b16':'SigLIP2 B/16','dinov2_small':'DINOv2 Small',
    'qwen3_vl':'Qwen3-VL API','qwen3_vl_instruct':'Qwen3-VL API + attribute instruct','tongyi_plus':'Tongyi Vision Plus 2026-03-06'}

def stats(v):return {'mean':statistics.mean(v),'std':statistics.stdev(v),'values':v}
def fmt(v):return f"{v['mean']*100:.2f} ± {v['std']*100:.2f}"

def main():
    out=Path('artifacts/results/v4');p=json.loads((out/'test_results.json').read_text())
    previous=json.loads(Path('artifacts/results/v3/test_results.json').read_text())
    groups=defaultdict(list)
    for r in p['results']+previous['results']:
        if r['arch']=='raw':groups[(r['dataset'],r['backend'],r['scale'])].append(r)
    summary=[]
    for (d,b,s),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['seed']);assert len(rows)==3
        summary.append(dict(dataset=d,backend=b,scale=s,lr=rows[0]['lr'],
            attribute_mAP=stats([r['metrics']['attributes']['macro']['mAP'] for r in rows]),
            joint_mAP=stats([r['metrics']['joint']['mAP'] for r in rows]),
            attribute_R1=stats([r['metrics']['attributes']['macro']['R@1'] for r in rows]),
            best_steps=[r['best_step'] for r in rows],trainable_parameters=rows[0]['trainable_parameters']))
    index={(r['dataset'],r['backend'],r['scale']):r for r in summary}
    frozen={}
    for r in p['frozen_baselines']+previous['frozen_baselines']:
        frozen[(r['dataset'],r['backend'])]=r
    order=['omnijev','base','flair',*MODELS]
    lines=['# Mainstream embedding comparisons (v4)','',
        'Full fixed v3 test galleries; attribute macro mAP (%). Supervised rows: three seeds, mean ± sample SD. No test-based hyperparameter selection. Native frozen embeddings and supervised heads are separate tracks.','',
        '## Frozen native features','',
        '| Model | Fashion | CUB | DTD | Native width per slot |', '|---|---:|---:|---:|---:|']
    for b in order:
        values=[100*frozen[(d,b)]['raw']['attributes']['macro']['mAP'] for d in DATASETS]
        width=MODELS[b]['width'] if b in MODELS else 1024 if b in ('omnijev','base') else 512
        lines.append(f"| {LABELS[b]} | "+' | '.join(f'{v:.2f}' for v in values)+f' | {width} |')
    values=[100*frozen[(d,'flair')]['global']['attributes']['macro']['mAP'] for d in DATASETS]
    lines.append('| FLAIR global | '+' | '.join(f'{v:.2f}' for v in values)+' | 512 |')
    lines+=['','## Equal training budget, full training data','',
        'Shared LN + Linear(256), 1,000 optimizer steps, batch 64, same loss, two learning rates and three seeds. Backbones, resolutions, pretraining data and service compute differ. Global features yield identical slots after a shared head.','',
        '| Model | Fashion | CUB | DTD |','|---|---:|---:|---:|']
    for b in order:lines.append(f"| {LABELS[b]} | "+' | '.join(fmt(index[(d,b,100)]['attribute_mAP']) for d in DATASETS)+' |')
    lines+=['','The head architecture and update budget are shared, but input widths differ, so trainable parameter counts are not equal.','',
        '| Model | Head trainable parameters |','|---|---:|']
    for b in order:lines.append(f"| {LABELS[b]} | {index[(DATASETS[0],b,100)]['trainable_parameters']} |")
    lines+=['','## Full-data joint retrieval','', '| Model | Fashion | CUB | DTD |','|---|---:|---:|---:|']
    for b in order:lines.append(f"| {LABELS[b]} | "+' | '.join(fmt(index[(d,b,100)]['joint_mAP']) for d in DATASETS)+' |')
    lines+=['','## Data scaling (attribute mAP)','', '| Dataset | Model | 25% train | 50% train | 100% train |','|---|---|---:|---:|---:|']
    for d in DATASETS:
        for b in order:lines.append(f'| {NAMES[d]} | {LABELS[b]} | '+' | '.join(fmt(index[(d,b,s)]['attribute_mAP']) for s in (25,50,100))+' |')
    lines+=['','## Controlled Qwen instruction comparison','',
        'Same API model and native dimension; fixed attribute instructions versus global image features. Differences below are instruct minus global, in percentage points. Seed-paired training differences do not establish statistical significance.','',
        '| Dataset | Frozen attribute mAP delta | Adapted attribute mAP delta |','|---|---:|---:|']
    contrasts=[]
    for d in DATASETS:
        native=100*(frozen[(d,'qwen3_vl_instruct')]['raw']['attributes']['macro']['mAP']-frozen[(d,'qwen3_vl')]['raw']['attributes']['macro']['mAP'])
        diff=stats([100*(x-y) for x,y in zip(index[(d,'qwen3_vl_instruct',100)]['attribute_mAP']['values'],index[(d,'qwen3_vl',100)]['attribute_mAP']['values'])])
        contrasts.append(dict(dataset=d,frozen_delta_pp=native,adapted_paired_delta_pp=diff))
        lines.append(f"| {NAMES[d]} | {native:+.2f} | {diff['mean']:+.2f} ± {diff['std']:.2f} |")
    lines+=['','## Full-data per-attribute mAP','', '| Dataset | Model | Attribute | mAP (%) |','|---|---|---|---:|']
    for d in DATASETS:
        for b in order:
            rows=groups[(d,b,100)]
            for key in rows[0]['metrics']['attributes']:
                if key=='macro':continue
                value=stats([r['metrics']['attributes'][key]['mAP'] for r in rows])
                lines.append(f'| {NAMES[d]} | {LABELS[b]} | {key} | {fmt(value)} |')
    lines+=['','## API accounting','',
        'Usage is measured from successful cached batches. Retries or interrupted uncached requests may additionally be billed; the provider invoice is authoritative. API latency was measured during overlapping tasks and is not a hardware-normalized speed benchmark.','']
    accounting=out/'api_accounting.json'
    if accounting.exists():
        acc=json.loads(accounting.read_text())
        lines+=['| API backend | Requests | Total tokens | Extra attempts |','|---|---:|---:|---:|']
        for b,r in acc.items():lines.append(f"| {LABELS[b]} | {r['requests']} | {r['total_tokens']} | {r['extra_attempts']} |")
    limited=sum(r['best_step']==1000 for r in p['results'])
    lines+=['',f'{limited}/{len(p["results"])} new selected checkpoints are best at the 1,000-step budget boundary. This is a fixed-budget comparison, not a convergence claim.',
        '', 'CUB excludes missing attributes and uses observed positive-set overlap. DTD is split 1 attribute retrieval, not the official ten-split classification benchmark. All test query coverage is retained in test_results.json. Qwen service aliases cannot be pinned to open weights; cached vectors, dates and parameters identify the measured service.','']
    Path('reports/RESULTS_V4.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json({'summary':summary,'instruction_contrasts':contrasts,'selection_fingerprint':p['selection_fingerprint']},out/'summary.json')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    colors=plt.cm.tab10.colors
    for ax,d in zip(axes,DATASETS):
        for i,b in enumerate(order):
            rs=[index[(d,b,s)] for s in (25,50,100)]
            ax.errorbar([25,50,100],[r['attribute_mAP']['mean']*100 for r in rs],
                yerr=[r['attribute_mAP']['std']*100 for r in rs],marker='o',markersize=3,color=colors[i],label=LABELS[b])
        ax.set(title=NAMES[d],xlabel='Training subset (%)',ylabel='Attribute mAP (%)',xticks=[25,50,100]);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.2,1,1));Path('artifacts/figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(f'artifacts/figures/v4_mainstream_scaling.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
