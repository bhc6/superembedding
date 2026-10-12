"""Evaluate all validation-sealed heads on held sources; no test selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.checkpoint import atomic_json,fingerprint
from superembedding.general_ablation import GeneralAblationHead
from scripts.train_general_ablation import read_bank,evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=Path(args.output);raw=(out/'selection_sealed.json').read_bytes();sealed=json.loads(raw)
    if len(sealed['results'])!=36 or len(sealed['selected'])!=6:raise ValueError('Incomplete validation grid')
    prov,rows,q,d,_=read_bank(args.features,{'val','ood'})
    if fingerprint(prov)!=sealed['feature_provenance']:raise ValueError('Feature selection mismatch')
    val=[i for i,r in enumerate(rows) if r['split']=='val'];ood=[i for i,r in enumerate(rows) if r['split']=='ood']
    baseline={'validation':evaluate(None,q,d,rows,val),'ood':evaluate(None,q,d,rows,ood),'dimension':4096}
    tested=[]
    for selected in sealed['selected']:
        if len(selected['runs'])!=3:raise ValueError('All three seeds required')
        for folder in selected['runs']:
            run=Path(folder);saved=torch.load(run/'best.pt',weights_only=True,map_location='cpu')
            result=json.loads((run/'result.json').read_text());config=result['config']
            if saved['fingerprint']!=fingerprint(config) or saved['config']!=config:raise ValueError('Checkpoint config mismatch')
            for path,digest in config['implementation_sha256'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Ablation implementation changed: '+path)
            if saved['step']!=result['best_step']:raise ValueError('Selected step mismatch')
            head=GeneralAblationHead(config['architecture']);head.load_state_dict(saved['model'],strict=True)
            validation=evaluate(head,q,d,rows,val)
            if abs(validation['macro_r1']-result['best_validation'])>1e-7:raise ValueError('Validation replay mismatch')
            tested.append({'architecture':config['architecture'],'seed':config['seed'],'lr':config['lr'],
                'step':saved['step'],'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
                'parameters':result['parameters'],'dimension':result['output_dimension'],
                'validation':validation,'ood':evaluate(head,q,d,rows,ood)})
    summaries=[]
    for arch in [r['architecture'] for r in sealed['selected']]:
        runs=[r for r in tested if r['architecture']==arch];scores=[r['ood']['macro_r1'] for r in runs]
        summaries.append({'architecture':arch,'mean_ood_r1':statistics.mean(scores),'seed_sd':statistics.stdev(scores),
            'parameters':runs[0]['parameters'],'dimension':runs[0]['dimension'],
            'mean_val_r1':statistics.mean(r['validation']['macro_r1'] for r in runs),
            'steps':[r['step'] for r in runs]})
    report={'selection_sha256':hashlib.sha256(raw).hexdigest(),'feature_provenance':fingerprint(prov),
        'baseline':baseline,'runs':tested,'summary':summaries,'validation_replay_passed':True,
        'scope':'Frozen pretrained OmniJev, 128 train images/source, one fixed supervision/image. OOD from upstream train sources, not official test.',
        'architecture_scope':'Mean-pool retains four branch computations; not a single-branch/global-backbone ablation. Output projections change parameter counts.'}
    atomic_json(report,'artifacts/results/v5/general_ablation_results.json')
    lines=['# 通用冻结特征的架构消融','',report['scope'],'',
        '36 次训练；每架构按三个种子的平均验证分数选择一个学习率，封存后评测来源留出。标准差是训练种子标准差。', '',
        '| 架构 | 参数 | 输出维度 | 验证 R@1 % | 来源留出 R@1 % ± SD | 选中步 |',
        '|---|---:|---:|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['architecture']} | {r['parameters']} | {r['dimension']} | {r['mean_val_r1']*100:.2f} | {r['mean_ood_r1']*100:.2f} ± {r['seed_sd']*100:.2f} | {r['steps']} |")
    lines += ['',f"未训练 raw concat（4096 维）来源留出 R@1：{baseline['ood']['macro_r1']*100:.2f}%。",'',
        '各来源候选及正例数量见 JSON；AI2D/ScienceQA 来源留出不是官方测试。step 0 入选不是训练收益。', '',
        '等维 output projection 以单位阵初始化，初始相似度等价，但增加参数；独立头同维版约四倍参数，另列近参数预算版。均值池化仍计算四个方向，不等于单分支 backbone。', '',
        '本实验冻结预训练 OmniJev，不检验长训 LoRA 后的架构排序；数据规模和固定单监督限制须保留，不据此宣称通用最优。','']
    Path('reports/GENERAL_ABLATION_V5.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
