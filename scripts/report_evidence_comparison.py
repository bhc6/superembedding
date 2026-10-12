"""Paired scene-cluster bootstrap for frozen evidence diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from superembedding.checkpoint import atomic_json


def paired_summary(values, reference, resamples):
    """Inputs contain per-scene rates, with all within-scene cases kept together."""
    values=np.asarray(values,dtype=float);reference=np.asarray(reference,dtype=float)
    means=values[resamples].mean(axis=1)
    delta=(values-reference)[resamples].mean(axis=1)
    return {'rate':float(values.mean()),'cluster_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),
            'minus_reference':float((values-reference).mean()),
            'paired_cluster_bootstrap_95':np.quantile(delta,[.025,.975]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['qwen3_vl','clip_b32','siglip2_b16','flair_global','flair_native'])
    ap.add_argument('--reference',default='qwen3_vl');a=ap.parse_args()
    out=Path('artifacts/results/v5');records={};hashes={};dataset=None
    for model in a.models:
        path=out/f'evidence_{model}.json';raw=path.read_bytes();obj=json.loads(raw)
        if dataset is not None and dataset!=obj['dataset_sha256']:raise ValueError('Dataset mismatch')
        dataset=obj['dataset_sha256'];hashes[model]=hashlib.sha256(raw).hexdigest()
        records[model]={r['id']:r for r in obj['cases']}
        if len(records[model])!=len(obj['cases']):raise ValueError('Duplicate case ID')
    reference=records[a.reference];ids=sorted(reference)
    for model,rows in records.items():
        if set(rows)!=set(ids):raise ValueError('Case coverage mismatch')
        for i in ids:
            if any(rows[i][k]!=reference[i][k] for k in ['scene_group','attribute']):raise ValueError('Case identity mismatch')
    scenes=sorted({r['scene_group'] for r in reference.values()})
    attrs=sorted({r['attribute'] for r in reference.values()})
    rng=np.random.default_rng(20261011);samples=rng.integers(0,len(scenes),size=(10000,len(scenes)))
    metrics={}
    for scope in attrs+['macro_all_attributes']:
        metric={}
        for kind in ['anchor_preference','robust_both_preference']:
            per_model={}
            for model,rows in records.items():
                values=[]
                for scene in scenes:
                    cases=[rows[i] for i in ids if rows[i]['scene_group']==scene
                           and (scope=='macro_all_attributes' or rows[i]['attribute']==scope)]
                    expected=len(attrs) if scope=='macro_all_attributes' else 1
                    if len(cases)!=expected:raise ValueError('Unbalanced scene/attribute coverage')
                    values.append(np.mean([r['anchor_margin']>0 if kind=='anchor_preference'
                        else min(r['anchor_margin'],r['nuisance_margin'])>0 for r in cases]))
                per_model[model]=values
            metric[kind]={model:paired_summary(v,per_model[a.reference],samples) for model,v in per_model.items()}
        metrics[scope]=metric
    result={'dataset_sha256':dataset,'result_sha256':hashes,'reference':a.reference,'scene_groups':len(scenes),
        'cases':len(ids),'bootstrap_replicates':len(samples),'seed':20261011,'metrics':metrics,
        'scope':'Exploratory diagnostic; percentile intervals are marginal, not multiplicity-adjusted; no architecture selection.',
        'dependence':'Resample complete scenes, preserving paired models and all four interventions.',
        'position_caveat':'Swaps both A/B positions; pooled macro is descriptive, not one homogeneous causal intervention.'}
    atomic_json(result,out/'evidence_comparison.json')
    lines=['# 冻结模型的成对证据诊断','',
        f'{len(scenes)} 个场景、{len(ids)} 个案例。预训练基线无任务训练。Qwen/CLIP/SigLIP/FLAIR global 对候选使用固定向量；FLAIR native 缓存图像 token，但每次查询重新条件池化，不属于固定向量检索。', '',
        '指标：原图和无关干扰图都排在关键证据错误图之前的比例。严格大于，平分不计成功。', '',
        '| 属性 | 模型 | 比例 % | 场景 bootstrap 95% CI | 相对 Qwen 差值 pp（成对 95% CI） |',
        '|---|---|---:|---:|---:|']
    for scope,metric in metrics.items():
        for model,m in metric['robust_both_preference'].items():
            lo,hi=m['cluster_bootstrap_95'];dl,dh=m['paired_cluster_bootstrap_95']
            lines.append(f"| {scope} | {model} | {100*m['rate']:.2f} | {100*lo:.2f}–{100*hi:.2f} | {100*m['minus_reference']:.2f} [{100*dl:.2f}, {100*dh:.2f}] |")
    lines+=['', '按场景进行 10,000 次成对重采样，保留每场景四属性和模型配对。区间是探索性的边际区间，未作多重比较校正。', '',
        '边界提示：SigLIP 数字项为 64/64，经验 percentile bootstrap 退化为 [100%, 100%]，不表示总体零不确定性；独立场景二项 Wilson 95% 区间为约 [94.34%, 100%]。', '',
        '位置干预同时交换 A/B，宏平均只作描述。单一合成渲染器、固定提示模板，不代表自然图像泛化、SOTA 或预训练数据无污染。CPU/API 时间不作横向延迟对比。', '',
        '是否包含训练后的 OmniJev，以表中模型及结果来源为准；缺少该模型时，本表不能说明项目模型优于这些基线。','']
    Path('reports/EVIDENCE_V5.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({s:{m:round(v['rate']*100,2) for m,v in r['robust_both_preference'].items()} for s,r in metrics.items()}))


if __name__=='__main__':main()
