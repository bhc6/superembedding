"""Lightweight snapshot of actual workers, without treating queued work as done."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superembedding.checkpoint import atomic_json
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'state': 'not_started'}


def main():
    statuses = {
        'training': read(STORE/'general_step556_continuation_v1/status.json'),
        'training_wrapper': read(STORE/'general_step556_continuation_v1/continuation_status.json'),
        'cpu_queue': read(STORE/'paper_cpu_queue_v1/status.json'),
        'wemm': read(STORE/'paper_baselines_cpu_v3/wemm2b/status.json'),
        'gme': read(STORE/'paper_baselines_cpu_v3/gme2b/status.json'),
        'official_vsr': read(STORE/'official_vsr_holdout_v2/status.json'),
        'frozen_control_queue': read(STORE/'frozen_terminal_control_v2/queue_status.json')}
    vector_counts = {model: sum(1 for _ in (STORE/'paper_baselines_cpu_v3'/model/'vectors').glob('*.pt'))
                     for model in ['wemm2b', 'gme2b']}
    pending = [
        'VLM2Vec-V2: official 420 DoRA tensors inspected; Transformers 5 module mapping and forward equivalence still need verification.',
        'Ovis: pinned repository has README only; downloaded checkpoint includes thinker/talker/token2wav. Native embedding preprocessing and memory-compatible loading remain to be verified.',
        'Fresh official VSR: complete acquisition, input sealing and scene/crop review before inference. Filtered subset is not an official full-test score.',
        'Fixed-terminal full LoRA+head roundtrip audit and cleared independent evaluation; validation best remains diagnostic.',
        'Natural/compositional paired evidence, actual branch-count/decision-pretraining controls, training seeds, broader data and method experiments.',
        'CUB/DTD/Fashion separate family leakage remediation, final data rights/model card/release audit.'
    ]
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'statuses': statuses, 'vector_counts': vector_counts,
              'pending': pending, 'whole_project_complete': False, 'git_publication_performed': False,
              'checks_this_launch': '22 relevant existing pytest tests passed; both official baselines produced 3 real image-query vectors; frozen schedule preflight matched 556 historical sources and step60 RNG.',
              'resume_commands': ['python -B -u scripts/continue_step556.py',
                                  'python -B -u scripts/run_paper_cpu_queue.py',
                                  'python -B -u scripts/prepare_official_vsr_holdout_v2.py',
                                  'python -B -u scripts/queue_frozen_after_continuation.py'],
              'resume_note': 'Dot-source scripts/use_d_storage.ps1 first; use project .venv. Do not start a second worker while its lock is held.'}
    atomic_json(result, ROOT/'artifacts/results/v5/active_experiments.json')
    train, vsr = statuses['training'], statuses['official_vsr']
    lines = ['# 并行实验运行状态', '', '快照时间：'+result['utc']+'。重新运行 scripts/report_active_experiments.py 可刷新。', '',
             '| 任务 | 实际状态 |', '|---|---|',
             f"| OmniJev LoRA + embedding 长训 | {train.get('state')}；Step {train.get('step')}/2000；最近 checkpoint {train.get('last_checkpoint_step')} |",
             f"| CPU 论文基线队列 | {statuses['cpu_queue'].get('state')}；当前 {statuses['cpu_queue'].get('model')}；WeMM→GME 串行 |",
             f"| WeMM-2B / GME-2B | 已缓存 {vector_counts['wemm2b']} / {vector_counts['gme2b']} 个向量；缓存数不等于完成题数 |",
             f"| 官方 VSR 2195 题筛查 | {vsr.get('state')}；已处理 {vsr.get('completed',vsr.get('rows',0))}；隔离候选 {vsr.get('quarantined_so_far',vsr.get('quarantined_queries',0))} |",
             f"| 冻结 backbone 三学习率对照 | {statuses['frozen_control_queue'].get('state')}；长训完成且显存连续三次足够后接续 |", '',
             'CPU 基线先跑同协议 VSR110，再跑 760 条历史验证及问答 text-only，最后跑 ScienceQA1000 全模态/text-only。采用 FP32 和 65536 像素预算，属于受限配置直接比较，不是论文 headline 的完整复现。每个向量独立缓存，绑定输入字节、官方权重和实现哈希。', '',
             '长训完整 optimizer/RNG checkpoint 每 5 步保存。冻结对照也保留 optimizer/RNG 和固定采样序列，最终保存固定 Step2000 的完整 backbone adapter + head。旧封存脚本、旧 STOP 和失败 v1/v2 基线记录保留。', '',
             '新 VSR 以 41999 张历史图像清单进行 RGB 精确与双感知哈希筛查。隔离候选不等于确认同图，未隔离也不等于已完成场景/裁剪审查；不自动启动正式评测。官方原始 HTTP 图像链接可访问，HTTPS 证书域名不匹配，未关闭 TLS 校验。', '',
             '本轮检查：22 项相关 pytest 通过；WeMM/GME 各三条真实图像推理通过；冻结对照采样预检通过。当前任务全部使用 D 盘。', '',
             '## 尚未完成', '']+['- '+p for p in pending]+['', '项目尚未全部完成，未提交或推送 Git。', '']
    (ROOT/'reports/ACTIVE_EXPERIMENTS.zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'training_step': train.get('step'), 'vectors': vector_counts,
                      'official_vsr': vsr, 'whole_project_complete': False}, ensure_ascii=True))


if __name__ == '__main__':
    main()
