"""Read-only interim external report; cannot change training or checkpoint selection."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from superembedding.checkpoint import atomic_json
from scripts.report_evidence_comparison import paired_summary


def main():
    store=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    seal=json.loads((store/'selected_external_step250/selection_sealed.json').read_text(encoding='utf-8'))
    tag='general_'+seal['export_sha256'][:12]+'_cpu'
    root=Path('artifacts/results/v5')
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    qa=read(f'scienceqa_{tag}.json');control=read(f'scienceqa_{tag}_text_only.json')
    transfer=read(f'transfer_{tag}.json');raw_transfer=read('transfer_omnijev_raw_cpu.json')
    models=['qwen3_vl','clip_b32','siglip2_b16','flair_global','flair_native','omnijev_raw_cpu',tag]
    records={m:read(f'evidence_{m}.json') for m in models}
    reference=records['omnijev_raw_cpu']
    ref={r['id']:r for r in reference['cases']};scenes=sorted({r['scene_group'] for r in ref.values()})
    attrs=sorted({r['attribute'] for r in ref.values()})
    rng=np.random.default_rng(20261011);samples=rng.integers(0,len(scenes),size=(10000,len(scenes)))
    metrics={}
    for attr in attrs+['macro_all_attributes']:
        values={}
        for name,result in records.items():
            rows={r['id']:r for r in result['cases']}
            if result['dataset_sha256']!=reference['dataset_sha256'] or set(rows)!=set(ref):raise ValueError('Evidence coverage mismatch')
            if any((r['scene_group'],r['attribute'])!=(ref[k]['scene_group'],ref[k]['attribute']) for k,r in rows.items()):
                raise ValueError('Evidence grouping mismatch')
            per_scene=[]
            for scene in scenes:
                cases=[r for r in rows.values() if r['scene_group']==scene and (attr=='macro_all_attributes' or r['attribute']==attr)]
                if len(cases)!=(len(attrs) if attr=='macro_all_attributes' else 1):raise ValueError('Unbalanced evidence cases')
                per_scene.append(np.mean([min(r['anchor_margin'],r['nuisance_margin'])>0 for r in cases]))
            values[name]=per_scene
        metrics[attr]={name:paired_summary(v,values['omnijev_raw_cpu'],samples) for name,v in values.items()}
    result={'selection':seal,'scienceqa_full':{k:v for k,v in qa.items() if k!='predictions'},
            'scienceqa_text_only':{k:v for k,v in control.items() if k!='predictions'},
            'urban1k':{tag:transfer['urban1k_common_999'],'omnijev_raw_cpu':raw_transfer['urban1k_common_999']},
            'evidence':metrics,'evidence_reference':'omnijev_raw_cpu',
            'artifact_sha256':{f'evidence_{m}.json':hashlib.sha256((root/f'evidence_{m}.json').read_bytes()).hexdigest() for m in models},
            'limitations':['Interim validation-selected checkpoint, not final model','CPU FP32 for trained and raw OmniJev; other baselines have native execution',
                           'Single synthetic renderer, 64 scene groups; not natural-image causal evidence','No test-based tuning permitted']}
    atomic_json(result,root/'selected_external_step250.json')
    lines=['# Step 250 封存模型的外部诊断','',
           '这是按首轮完整验证封存的中途 checkpoint，不是 2000 步训练最终模型；结果只作诊断，不用于挑选 checkpoint 或架构。完整 LoRA＋head 使用与评测一致的加载器。','',
           f"官方 MMEB ScienceQA（1000 查询）：完整输入 {qa['hit@1']*100:.2f}%；text-only {control['hit@1']*100:.2f}%。相同候选向量，成对差值与区间见 JSON。",'',
           '| Urban1k 999 子集 | 文搜图 R@1 % | 图搜文 R@1 % |','|---|---:|---:|']
    for m,r in result['urban1k'].items():lines.append(f"| {m} | {r['text_to_image_R@1']*100:.2f} | {r['image_to_text_R@1']*100:.2f} |")
    lines+=['','| 证据属性 | 模型 | 原图及干扰图均优于证据改变图 % | 相对 raw 差值 pp [95% CI] |','|---|---|---:|---:|']
    for attr,values in metrics.items():
        for name,r in values.items():
            lo,hi=r['paired_cluster_bootstrap_95']
            lines.append(f"| {attr} | {name} | {r['rate']*100:.2f} | {r['minus_reference']*100:.2f} [{lo*100:.2f}, {hi*100:.2f}] |")
    lines+=['','证据区间按完整场景做 10000 次成对重采样，四属性同时保留，未做多重比较校正。位置项同时交换两个物体；64/64 等边界比例的经验 bootstrap 可退化，不意味着总体零不确定性。','',
            '无关干扰不变在这里指给定查询的检索偏好保持，不能要求通用向量完全相等：对当前查询无关的物体属性，可能是另一个查询的关键证据。固定向量在所有查询间复用，测试时没有重新按查询编码候选。','',
            '单一合成渲染风格不等于真实场景机制已证实；Urban1k 强基线接近饱和。来源留出、污染检查和这些诊断均不能单独支持 SOTA 或发表级结论。','']
    Path('reports/SELECTED_EXTERNAL_STEP250.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote interim frozen external report')


if __name__=='__main__':main()
