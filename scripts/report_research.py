"""Generate auditable tables/plots from all sealed v3 test evaluations."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES={'fashion_expanded':'Fashion','cub_attributes':'CUB attributes','dtd':'DTD textures'}

def meanstd(values): return {'mean':statistics.mean(values),'std':statistics.stdev(values),'values':values}

def main():
    out=ROOT/'artifacts/results/v3'; payload=json.loads((out/'test_results.json').read_text())
    assert len(payload['results'])==216
    grouped=defaultdict(list)
    for r in payload['results']: grouped[(r['dataset'],r['backend'],r['scale'],r['arch'])].append(r)
    summary=[]
    for (dataset,backend,scale,arch),rows in sorted(grouped.items()):
        rows.sort(key=lambda r:r['seed']); assert [r['seed'] for r in rows]==[17,42,91]
        row={'dataset':dataset,'backend':backend,'scale':scale,'arch':arch,
             'train_count':json.loads((ROOT/'data'/dataset/'metadata.json').read_text())['scales'][str(scale)],
             'lr':rows[0]['lr'],'output_dimension':rows[0]['output_dimension'],
             'trainable_parameters':rows[0]['trainable_parameters'],'best_steps':[r['best_step'] for r in rows]}
        for key in ('mAP','R@1','R@5'):
            row['attribute_'+key]=meanstd([r['metrics']['attributes']['macro'][key] for r in rows])
            row['joint_'+key]=meanstd([r['metrics']['joint'][key] for r in rows])
        row['attribute_details']={k:{m:meanstd([r['metrics']['attributes'][k][m] for r in rows])
                                     for m in ('mAP','R@1','R@5')} for k in rows[0]['metrics']['attributes'] if k!='macro'}
        row['coverage']={k:{s:v[s] for s in ('eligible_queries','observed_queries','total_queries')}
                         for k,v in rows[0]['metrics']['attributes'].items() if k!='macro'}
        if dataset=='dtd': row['joint_texture_overlap_mAP']=meanstd([r['metrics']['joint_texture_overlap']['mAP'] for r in rows])
        summary.append(row)
    index={(r['dataset'],r['backend'],r['scale'],r['arch']):r for r in summary}
    deltas=[]
    for dataset in NAMES:
        for backend in ('omnijev','flair'):
            base=index[(dataset,backend,100,'joint_concat')]
            for arch in ('joint_out_equal','joint_out_small'):
                variant=index[(dataset,backend,100,arch)]
                deltas.append({'dataset':dataset,'backend':backend,'comparison':arch+' minus joint_concat',
                    'joint_mAP_percentage_point_delta':meanstd([100*(a-b) for a,b in zip(variant['joint_mAP']['values'],base['joint_mAP']['values'])]),
                    'attribute_mAP_percentage_point_delta':meanstd([100*(a-b) for a,b in zip(variant['attribute_mAP']['values'],base['attribute_mAP']['values'])])})
    report={'selection_fingerprint':payload['selection_fingerprint'],'summary':summary,'projection_deltas':deltas,
            'frozen_baselines':payload['frozen_baselines'],'random_ranking_expectation':payload['random_ranking_expectation'],
            'uncertainty':'sample standard deviation across three training seeds; not statistical significance or dataset-bootstrap confidence intervals'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    def fmt(v): return f"{v['mean']*100:.2f} ± {v['std']*100:.2f}"
    lines=['# V3 multi-domain test results','',
           'All learning rates and checkpoints selected on validation before test. Mean ± sample SD over seeds 17/42/91. Metrics are percentages. DTD is official split 1 image retrieval, not 10-split classification.','',
           '| Dataset | Backend | Architecture | Train N | Attribute mAP | Joint mAP | Attribute R@1 | Global dim | Best steps |',
           '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary:
        if r['scale']!=100: continue
        lines.append(f"| {NAMES[r['dataset']]} | {r['backend']} | {r['arch']} | {r['train_count']} | {fmt(r['attribute_mAP'])} | {fmt(r['joint_mAP'])} | {fmt(r['attribute_R@1'])} | {r['output_dimension']} | {r['best_steps']} |")
    lines+=['','## Data scaling at a fixed optimizer-update budget','',
            '| Dataset | Backend | Architecture | Train N | Attribute mAP | Joint mAP |','|---|---|---|---:|---:|---:|']
    for r in summary:
        if r['arch']=='raw' or r['backend']=='omnijev' and r['arch'].startswith('joint'):
            lines.append(f"| {NAMES[r['dataset']]} | {r['backend']} | {r['arch']} | {r['train_count']} | {fmt(r['attribute_mAP'])} | {fmt(r['joint_mAP'])} |")
    lines+=['','## Output projection paired differences','',
            'Differences pair the same training seeds. No significance claim is made from three seeds.','',
            '| Dataset | Backend | Comparison | Joint mAP delta (pp) |','|---|---|---|---:|']
    for r in deltas:
        v=r['joint_mAP_percentage_point_delta']; lines.append(f"| {NAMES[r['dataset']]} | {r['backend']} | {r['comparison']} | {v['mean']:+.2f} ± {v['std']:.2f} |")
    lines+=['','## Frozen and random-ranking controls','',
            '| Dataset | Control | Attribute mAP | Joint mAP |','|---|---|---:|---:|']
    for dataset,metrics in payload['random_ranking_expectation'].items():
        lines.append(f"| {NAMES[dataset]} | Uniform random ranking (analytic expectation) | {100*metrics['attributes']['macro']['mAP']:.2f} | {100*metrics['joint']['mAP']:.2f} |")
    for r in payload['frozen_baselines']:
        for variant in ('raw','global'):
            if variant not in r: continue
            metrics=r[variant]
            lines.append(f"| {NAMES[r['dataset']]} | {r['backend']} frozen {variant} | {100*metrics['attributes']['macro']['mAP']:.2f} | {100*metrics['joint']['mAP']:.2f} |")
    lines+=['','## Query coverage','',
            '| Dataset | Attribute | Queries with positives | Observed queries | All queries |','|---|---|---:|---:|---:|']
    for dataset in NAMES:
        for key,counts in index[(dataset,'omnijev',100,'raw')]['coverage'].items():
            lines.append(f"| {NAMES[dataset]} | {key} | {counts['eligible_queries']} | {counts['observed_queries']} | {counts['total_queries']} |")
    lines+=['','Compressed global vectors do not replace slots for attribute-selective retrieval. Backbone and image resolution differ across methods; these are same-data/protocol comparisons, not equal-compute architecture comparisons. CUB missing-attribute pairs are excluded and query coverage is recorded in summary.json.','']
    (ROOT/'reports/RESULTS_V3.md').write_text('\n'.join(lines),encoding='utf-8')
    figures=ROOT/'artifacts/figures'; figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,dataset in zip(axes,NAMES):
        for backend,arch,label in [('omnijev','raw','OmniJev raw'),('omnijev','joint_concat','OmniJev concat'),
                                  ('omnijev','joint_out_small','OmniJev compressed'),('base','raw','LoRA disabled'),('flair','raw','FLAIR raw')]:
            rows=[index[(dataset,backend,s,arch)] for s in (25,50,100)]
            ax.errorbar([r['train_count'] for r in rows],[100*r['attribute_mAP']['mean'] for r in rows],
                        yerr=[100*r['attribute_mAP']['std'] for r in rows],marker='o',markersize=4,label=label)
        ax.set_title(NAMES[dataset]); ax.set_xlabel('Training images'); ax.set_ylabel('Attribute mAP (%)'); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(figures/f'v3_data_scaling.{ext}',dpi=200)
    plt.close(fig)
    seal=json.loads((out/'selection_sealed.json').read_text())
    selected=defaultdict(list)
    for item in seal['selected']:
        if item['scale']==100: selected[(item['dataset'],item['backend'],item['arch'])].append(item)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,dataset in enumerate(NAMES):
        for row,archs in enumerate((['raw','gated_frozen','gated_tuned','residual'],['joint_concat','joint_out_equal','joint_out_small'])):
            ax=axes[row,col]
            for arch in archs:
                items=selected[(dataset,'omnijev',arch)]
                ys=[[100*h['score'] for h in item['validation']['history']] for item in items]
                avg=[statistics.mean(y[i] for y in ys) for i in range(4)]
                ax.plot([250,500,750,1000],avg,marker='o',markersize=3,label=arch)
            ax.set_title(NAMES[dataset]+(' / attribute objective' if row==0 else ' / joint objective'))
            ax.set_xlabel('Optimizer updates');ax.set_ylabel('Mean validation selection score (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(figures/f'v3_validation_curves.{ext}',dpi=200)
    plt.close(fig)
    print(f'Wrote {len(summary)} aggregate rows and {len(deltas)} paired projection comparisons.')

if __name__=='__main__': main()
