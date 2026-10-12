"""Report complete, provenance-aligned controls without treating pending jobs as results."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json


def main():
    folder=Path('artifacts/results/v5/validation_controls')
    root=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    majority=json.loads((folder/'majority.json').read_text(encoding='utf-8'))
    protocol=majority['protocol'];rows=[]
    for path in sorted(folder.glob('*.json')):
        if path.name in {'majority.json','summary.json'}:continue
        r=json.loads(path.read_text(encoding='utf-8'))
        if 'full' not in r:continue
        prov=r['provenance']
        if prov['protocol']!=protocol:raise ValueError('Cross-protocol result: '+path.name)
        label=path.stem
        rows.append({'label':label,'result_file':path.as_posix(),'macro_r1':r['full']['macro_r1'],
                     'per_source':r['full']['per_source'],
                     'qa_full':r.get('qa_full'),'qa_text_only':r.get('qa_text_only'),
                     'qa_paired_delta':r.get('qa_paired_delta'),
                     'precision':prov.get('dtype','Provider-managed' if label.startswith('qwen_api') else 'float32'),
                     'limitation':r.get('limitation')})
    history=[json.loads(s) for s in (root/'general_stage1_v51/history.jsonl').read_text(encoding='utf-8').splitlines()]
    validations=[r for r in history if 'validation' in r]
    sources=sorted(majority['protocol'] and rows[0]['per_source'])
    lines=['# 同协议基线与问答捷径检查','',
           '完整验证：760 张图像，每图一条按 ID 固定选取的监督。每来源使用该来源全部验证目标作候选，保留重复答案并沿用训练正例规则；指标为 8 来源等权 R@1。问答控制只覆盖 ChartQA、CLEVR、DocVQA、VSR 四来源。','',
           '| 模型 / 执行路径 | 8 来源平均 % | '+ ' | '.join(sources)+' |',
           '|---|---:|'+'---:|'*len(sources)]
    for row in rows:
        lines.append('| '+row['label']+' | '+f"{row['macro_r1']*100:.2f}"+' | '+' | '.join(f"{row['per_source'][s]['R1']*100:.2f}" for s in sources)+' |')
    for entry in validations:
        v=entry['validation']
        if v['scope']!='All held validation images per source':continue
        lines.append(f"| 训练日志 Step {entry['step']} / CUDA BF16 | {v['macro_r1']*100:.2f} | "+' | '.join(f"{v['per_source_r1'][s]*100:.2f}" for s in sources)+' |')
    lines += ['', 'raw_cached / step0_reconstructed_cpu 使用冻结 CPU FP32 特征；不能直接把它们与训练日志的差当作精度受控的训练增益。CUDA 对照完成后单列。Step 0 原始文件已被覆盖，现在是按原种子、原构造器重建；RNG 与历史断点一致和参数差检查通过，但没有原始 Step 0 哈希，不能声称找回了原件。','',
              'Qwen API 是供应商可变别名，不等同于本地 2B。qwen2b 使用官方固定权重、默认指令及末 token 池化，图像限制 65536 pixels；这不是官方默认高分辨率设置。模型间原生 tokenizer/图像处理不同，同协议指相同数据、查询、候选、正例与指标。','',
              '## 多数类与 text-only','',
              '| 模型 / 控制 | 四问答来源平均 % | ChartQA % | CLEVR % | DocVQA % | VSR % |',
              '|---|---:|---:|---:|---:|---:|']
    qa_sources=['chartqa','clevr','docvqa','vsr']
    for name,r in majority['controls'].items():
        lines.append(f"| {name} | {r['macro_r1']*100:.2f} | "+' | '.join(f"{r['per_source'][s]['R1']*100:.2f}" for s in qa_sources)+' |')
    for row in rows:
        for mode in ['qa_full','qa_text_only']:
            r=row.get(mode)
            if r:
                lines.append(f"| {row['label']} / {mode} | {r['macro_r1']*100:.2f} | "+' | '.join(f"{r['per_source'][s]['R1']*100:.2f}" for s in qa_sources)+' |')
    lines += ['', '多数答案只从训练记录统计，每训练图固定一条监督；同频按答案字典序，未见问题类型回退到来源多数类。多数答案不在验证候选中则计错，不按验证标签重新选择。按问题类型的结果使用仅看问题文字的固定规则，不冒充官方类型。text-only 只删除查询图像，使用原封不动的候选向量。','',
              '## 问题类型明细','', '| 模型 | 来源 | 问题类型 | n | 完整输入 % | text-only % |', '|---|---|---|---:|---:|---:|']
    for row in rows:
        if not row['qa_full']:continue
        for source,r in row['qa_full']['per_source'].items():
            for typ,metric in r['per_question_type'].items():
                control=row['qa_text_only']['per_source'][source]['per_question_type'][typ]
                lines.append(f"| {row['label']} | {source} | {typ} | {metric['queries']} | {metric['R1']*100:.2f} | {control['R1']*100:.2f} |")
    lines += ['', '小样本类型仅作描述性分析。每来源完整输入减 text-only 的成对 bootstrap 区间在对应 JSON 中；按查询重采样，不把共享答案视为独立任务，不做多重检验后的显著性宣称。','',
              '## 外部测试与机制评测的锁定流程','',
              '当前 Step 250 是首轮完整验证选出的中途 checkpoint，已封存完整 LoRA＋head 并验证重载一致性。最终模型仍由预定 2000 步训练的验证分数选择。最终器先重跑同协议完整输入/text-only，再评测官方 MMEB ScienceQA、Urban1k 共同子集和成对证据诊断；不依外部结果换 checkpoint。','',
              '外部数据已做 RGB/dHash 和感知哈希近重复筛查，不能保证穷尽预训练污染。已有合成证据集只有一种渲染风格，不能替代自然图像、未见组合的机制证据；还不能宣称达到发表水平。','',
              '## 执行状态','']
    statuses={}
    for name,path in [('priority_gpu',root/'priority_validation/status.json'),
                      ('qwen2b',root/'validation_controls/qwen2b_cpu/status.json'),
                      ('training',root/'general_stage1_v51/status.json')]:
        if path.exists():
            statuses[name]=json.loads(path.read_text(encoding='utf-8'))
            lines += [f'### {name}','','```json',json.dumps(statuses[name],ensure_ascii=False,indent=2),'```','']
    atomic_json({'protocol':protocol,'results':rows,'training_validation':[{'step':r['step'],'validation':r['validation']} for r in validations],
                 'statuses':statuses,'test_based_selection':False},folder/'summary.json')
    Path('reports/VALIDATION_CONTROLS_V5.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote reports/VALIDATION_CONTROLS_V5.zh.md')


if __name__=='__main__':main()
