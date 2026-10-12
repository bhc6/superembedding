"""Generate current V5 evidence report without treating background work as complete."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.checkpoint import atomic_json

def main():
    out=Path('artifacts/results/v5');root=Path('runs/v5/urban1k')
    ann=json.loads(Path('C:/Users/123/.cache/superembedding/v5/urban1k/annotations/annotations.json').read_text(encoding='utf-8'))['annotations']
    kept=[i for i,r in enumerate(ann) if r['caption'].strip()];results=[]
    for model in ['flair','clip_b32','siglip2_b16','qwen3_vl']:
        obj=torch.load(root/model/'scores.pt',weights_only=True)
        if isinstance(obj,torch.Tensor):obj={'native':obj}
        for mode,s in obj.items():
            s=s[kept][:,kept];n=len(kept)
            t=(s.T.argmax(-1)==torch.arange(n)).float();im=(s.argmax(-1)==torch.arange(n)).float()
            results.append({'model':model,'mode':mode,'text_to_image_R1':t.mean().item(),'image_to_text_R1':im.mean().item(),'n':n})
    atomic_json({'excluded':[i for i in range(len(ann)) if i not in kept],
        'reason':'Official empty caption; all models share same 999 candidates/queries','results':results},out/'urban1k_common_subset.json')
    lines=['# V5 进展与实测结果（未完成全部研究）','',f"生成于 {datetime.now(timezone.utc).isoformat()}。",'',
           '所有分数均为百分数。后台运行不等于训练完成，启动验证也不等于检索能力提高。本阶段未提交/发布 GitHub。','',
           '## 完整官方 MMEB ScienceQA 子任务','',
           '1000 查询，每个查询原始 736 候选，Hit@1。官方 TIGER-Lab/MMEB-eval 固定 revision；这不是完整 MMEB 排名。','',
           '| 本机实测 | Hit@1 | 95% Wilson 区间 |','|---|---:|---:|']
    for backend in ['qwen3_vl','clip_b32','siglip2_b16','omnijev_raw']:
        r=json.loads((out/f'scienceqa_{backend}.json').read_text())
        lines.append(f"| {backend} | {r['hit@1']*100:.2f} | {r['wilson_95'][0]*100:.2f}–{r['wilson_95'][1]*100:.2f} |")
    lines+=['','CLIP/SigLIP 为原生图文向量等权融合的启发式基线；OmniJev raw 为固定方向原始 hidden 归一化拼接，没有训练 embedding projection。Qwen API 原生支持混合图文；其 alias 不能冒充公开 2B 或 8B checkpoint。',
            '', 'OmniJev raw 的低分说明直接拿决策 hidden 做通用跨模态余弦检索没有自动得到对齐空间；不能因此否定训练后迁移假设，也不能隐去这个负结果。',
            '', '## ScienceQA 模态干预','']
    controls_path=out/'scienceqa_modality_controls.json'
    if controls_path.exists():
        controls=json.loads(controls_path.read_text())
        lines += ['同一组 1000 查询、同一冻结候选向量库；不重训。错配图片为固定循环位移，8 条同期完整输入 API 漂移探针通过。', '',
                  '| 输入 | Hit@1 | 完整输入优势（百分点） | 成对 bootstrap 95% 区间 |', '|---|---:|---:|---:|']
        for name,m in controls['metrics'].items():
            lo,hi=m['paired_bootstrap_95']
            lines.append(f"| {name} | {m['hit@1']*100:.1f} | {m['full_minus_control']*100:.1f} | {lo*100:.1f}–{hi*100:.1f} |")
        lines += ['', '该差异支持正确图像对分数有贡献，不证明模型推理能力或无预训练污染。区间按查询重采样，未校正重复答案/图像间依赖。']
    lines += ['', '## FLAIR 论文官方权重推理复现','',
            '完整 Urban1k 1000 图像/1000 描述，官方注释、tokenizer、图像 transform 和评分函数。float32 推理，官方入口默认 AMP；不是从头重训 FLAIR。','',
            '| FLAIR | 文搜图 R@1 | 图搜文 R@1 |','|---|---:|---:|']
    flair=json.loads((out/'urban1k_flair.json').read_text())
    for mode,metrics in flair['metrics'].items():lines.append(f"| {mode} | {metrics['text_to_image_R@1']*100:.2f} | {metrics['image_to_text_R@1']*100:.2f} |")
    eq=json.loads((root/'flair/official_equivalence.json').read_text())
    lines += ['',f"首样本 native 分数与官方 `compute_similarity_scores_attn_pool` 最大绝对差 {eq['max_abs_score_error']:.3g}。原生 FLAIR 需要每个查询对图像 token 做条件池化，其在线成本与固定缓存向量不同。",'',
              '## API 与本地模型共同协议：Urban1k 999 非空子集','',
              '官方注释的 index 548 / 592.jpg 是空描述，API 拒绝空文本。以下对所有模型同时排除此图像及描述，不能与上面的 1000 条协议混写。','',
              '| 模型 / 模式 | 文搜图 R@1 | 图搜文 R@1 |','|---|---:|---:|']
    for r in results:lines.append(f"| {r['model']} / {r['mode']} | {r['text_to_image_R1']*100:.2f} | {r['image_to_text_R1']*100:.2f} |")
    lines+=['','强 API 基线接近饱和，因此 Urban1k 只保留为兼容性诊断，不承担本项目主创新结论。预训练污染未知。',
            '', '## 可发布的成对证据诊断 v1','',
            '64 个原始合成场景、256 个属性干预案例、384 张原创 PNG；CC0 图像/标注，生成器随仓库开源。仅作诊断，不进入训练。每图只编码一次，跨查询复用。','',
            '| Qwen API | 原图优于证据改变图 | 原图与干扰图都优于证据改变图 |','|---|---:|---:|']
    r=json.loads((out/'evidence_qwen3_vl.json').read_text())
    for k,v in r['metrics'].items():lines.append(f"| {k} | {v['anchor_preference']*100:.2f} | {v['robust_both_preference']*100:.2f} |")
    lines += ['', '位置干预同时交换 A/B 的左右位置，应与颜色、形状、数字分开分析；当前只用一种绘图风格和一个查询模板。每四条共享一个场景，不能视为独立样本。低空间分数是诊断线索，不是新颖性或真实应用优势证明。',
              '', '同协议 CLIP / SigLIP 及场景成对 bootstrap 结果见 `EVIDENCE_V5.zh.md`。若报告存在，该表包括不训练的本地对照，不能只按 Urban1k 分数认定 Qwen 在所有证据任务上最强。',
              '', '## 新一代模型官方报告（非本机复现）','',
              '| 上游提交 | 190任务缺失计零均值 | 数值覆盖任务 |','|---|---:|---:|']
    audit=json.loads((out/'reported_leaderboard_audit.json').read_text(encoding='utf-8'))
    for r in audit['models'][:6]:lines.append(f"| {r['metadata'].get('model_name')} | {r['overall_zero_filled']:.2f} | {r['coverage']}/190 |")
    lines += ['',f"固定榜单 revision `{audit['revision']}`。这是上游提交的审计汇总，不是本机 SOTA 复现；舍入顺序与官方显示可能差 0.01。WeMM 已用 Qwen3.5，因此这一 backbone 选择本身不新。",'',
              '## 后台训练快照','']
    active_path=Path('runs/v5/active_general_run.json')
    active=json.loads(active_path.read_text()) if active_path.exists() else {
        'run':'C:/Users/123/.cache/superembedding/v5/general_stage1',
        'data':'C:/Users/123/.cache/superembedding/v5/data'}
    train=Path(active['run'])
    data=json.loads((Path(active['data'])/'audit.json').read_text());lines += [f"活动运行：`{train}`。",'',f"数据：{data['split_counts']}；{data['supervision_pairs']} 条图文/问答监督（包括验证与留出）。8 个训练来源。排除 {len(data['test_overlap_excluded'])} 条与 ScienceQA/Urban1k 完全 RGB 或完全 dHash 匹配记录，清单 SHA256 `{data['manifest_sha256']}`。",'']
    if (train/'migration.json').exists():
        migration=json.loads((train/'migration.json').read_text())
        lines += [f"第 {migration['step']} 步、首次验证前显式迁移：隔离 {migration['quarantined_held_records']} 条验证近重复候选；训练记录/顺序、权重/优化器/RNG 和采样队列保留。原运行、断点及代码快照保留。验证改为全部 760 条（每图固定一条监督），不再使用每来源 16 条小监测集。",'',
                  '近重复筛查采用 pHash≤6 且 dHash≤6。人工确认 ChartQA/Chart2Text 至少一对内容相同，另有界面近重复；部分只是相同图表模板，保守隔离可能改变验证分布。87 对候选对应 41 个验证记录；外部完整 ScienceQA/Urban1k 未筛出候选，但该筛查不证明穷尽去污染。','']
    if (train/'status.json').exists():
        status=json.loads((train/'status.json').read_text());atomic_json(status,out/'training_status_snapshot.json')
        lines += ['```json',json.dumps(status,ensure_ascii=False,indent=2),'```','']
    ablation=Path('C:/Users/123/.cache/superembedding/v5/general_ablation')
    if (ablation/'pipeline_status.json').exists():
        lines += ['## 通用冻结特征架构对照','',
            '6 结构 × 3 seeds × 2 学习率，共 36 次；每来源 128 张训练图，完整验证。CPU 原始 OmniJev 特征，和 LoRA 长训独立。协议见 `GENERAL_ABLATION_PROTOCOL_V5.zh.md`。', '']
        snapshot={'pipeline':json.loads((ablation/'pipeline_status.json').read_text())}
        features=ablation.parent/'general_ablation_features'
        if (features/'status.json').exists():snapshot['features']=json.loads((features/'status.json').read_text())
        atomic_json(snapshot,out/'general_ablation_status_snapshot.json')
        lines += ['```json',json.dumps(snapshot,ensure_ascii=False,indent=2),'```','']
        if (out/'general_ablation_results.json').exists():
            lines += ['结果见 `GENERAL_ABLATION_V5.zh.md`；参数量、维度和计算路径不同处单独列明，不将本组称为完整通用 LoRA 架构最优性证明。','']
        else:lines += ['尚无封存后的来源留出结果，不能报告哪种结构获胜。','']
    for path in sorted(out.glob('transfer_*.json')):
        result=json.loads(path.read_text());lines += ['',f"### 已完成的冻结迁移评测：{result['backend']}",'',
            '```json',json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2),'```','']
    for path in sorted(out.glob('scienceqa_general_*.json')):
        result=json.loads(path.read_text());lines += ['',f"冻结模型 `{result['backend']}` ScienceQA Hit@1：{result['hit@1']*100:.2f}%。包含中途诊断和 text-only 控制时应分别阅读，不能自动视作最终模型结果；checkpoint 由验证而非本测试选择。",'']
    lines += ['配置和操作说明见 `PROTOCOL_V5.zh.md`、`RUNNING_V5.zh.md`。原子 last/best 保存优化器/RNG，固定里程碑保存权重。已有 V3/V4 checkpoint 全部保留。',
              '', '同协议 Step 0/raw/主流基线及问答捷径结果见 `VALIDATION_CONTROLS_V5.zh.md`；原始 Step 0 已覆盖，明确区分重建权重。GPU 优先评测时长训会在完整断点暂停，并由有限任务驱动器恢复。',
              '', '## 未完成项','',
              '训练是否完成及是否已有最终冻结评测，以上方状态和对应结果文件为准；没有结果文件就不能报告新模型主结果。磁盘扩容及迁移后，官方模型下载队列已启动；VLM2Vec/GME/WeMM/Ovis 尚未完成本机推理复现。下载不等于评测完成。没有完整 MMEB-V3、音视频能力结论、多种子确认性实验或新架构的完整参数/计算匹配消融。下一阶段须用未见任务检验空间/证据敏感性，并审计天然负例和近重复、各来源许可，才能讨论发表。','']
    migration_path=out/'storage_migration.json'
    if migration_path.exists():
        migration=json.loads(migration_path.read_text())
        lines += ['## D 盘存储','',f"实际数据、断点和模型缓存在 `D:/DEMO/superembedding-storage`；已迁移 {migration['total_bytes']/1e9:.2f} GB，C 盘旧路径为目录链接。详见 `STORAGE_D.zh.md`。",'']
    downloads=Path('D:/DEMO/superembedding-storage/official_models/status.json')
    if downloads.exists():
        lines += ['官方模型下载状态（不代表推理复现完成）：','```json',downloads.read_text(),'```','']
    Path('reports/RESULTS_V5_PROGRESS.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote reports/RESULTS_V5_PROGRESS.zh.md')

if __name__=='__main__':main()
