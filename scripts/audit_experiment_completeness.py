"""Read-only numerical replay and coverage audit; never selects a model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.checkpoint import atomic_json, fingerprint
from superembedding.validation_protocol import load_protocol, score_vectors, majority_controls

STORE=Path('D:/DEMO/superembedding-storage/superembedding/v5')
RESULTS=Path('artifacts/results/v5')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def check_predictions(items,result):
    ids=[r['id'] for r in items];preds=result['predictions']
    require([p['id'] for p in preds]==ids,'Missing, duplicated or reordered query predictions')
    lookup={r['id']:r for r in items}
    for p in preds:
        require(p['candidate_id'] in lookup,'Unknown candidate')
        a=lookup[p['id']];b=lookup[p['candidate_id']]
        require(a['source']==b['source'],'Cross-source candidate')
        def keys(k):return set(k) if isinstance(k,(tuple,list)) else {k}
        require(bool(p['correct'])==bool(keys(a['key']) & keys(b['key'])),'Prediction relevance mismatch')
    for source,r in result['per_source'].items():
        source_ids={r['id'] for r in items if r['source']==source}
        hits=[p['correct'] for p in preds if p['id'] in source_ids]
        require(len(hits)==r['queries'] and abs(sum(hits)/len(hits)-r['R1'])<1e-8,'Source metric mismatch')
        require(sum(t['queries'] for t in r['per_question_type'].values())==len(hits),'Question-type coverage mismatch')
    require(set(result['per_source'])=={r['source'] for r in items},'Missing source')
    require(abs(sum(r['R1'] for r in result['per_source'].values())/len(result['per_source'])-result['macro_r1'])<1e-8,'Macro score mismatch')


def main():
    torch.set_num_threads(2)
    protocol,items,all_rows=load_protocol(STORE/'data_v51')
    qa=[r for r in items if r['task']!='caption'];qa_ids=[i for i,r in enumerate(items) if r['task']!='caption']
    checks=[];failures=[];pending=[]
    def check(name,fn):
        try:
            detail=fn();checks.append({'check':name,'passed':True,'detail':detail})
            print(json.dumps({'check':name,'passed':True}),flush=True)
        except Exception as exc:
            failures.append({'check':name,'error':type(exc).__name__,'message':str(exc)})
            print(json.dumps({'check':name,'passed':False,'message':str(exc)}),flush=True)
    def training():
        run=STORE/'general_stage1_v51';config=read(run/'config.json')
        require(config['data_sha256']==protocol['data_sha256'],'Training data changed')
        for path,sha in config['implementation_sha256'].items():require(digest(path)==sha,'Live training code changed: '+path)
        rows=[]
        for folder in [STORE/'general_stage1',run]:
            rows.extend(json.loads(s) for s in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines())
        steps=[r['step'] for r in rows]
        require(steps==list(range(1,max(steps)+1)),'Training history has gaps, duplicate or reordered steps')
        require(all(r['gradient_replay_max_error']<=5e-4 for r in rows),'Gradient replay failure')
        validations=[r for r in rows if 'validation' in r]
        expected=list(range(250,max(steps)+1,250))
        require([r['step'] for r in validations]==expected,'Missing scheduled full validation')
        selected=max(validations,key=lambda r:r['validation']['macro_r1'])
        snapshots={}
        for name in ['last.pt','best.pt','step_000500_weights.pt']:
            path=run/name
            if not path.exists():
                require(name!='step_000500_weights.pt' or max(steps)<500,'Missing immutable step500 weights');continue
            # Atomic replacement means each open reads a complete old or new file.
            saved=torch.load(path,map_location='cpu',weights_only=True)
            require(saved['config']==config,'Checkpoint config changed')
            require(len(saved['model'])==304,'Missing LoRA/head tensors')
            require(all(torch.isfinite(v).all() for v in saved['model'].values()),'Nonfinite model tensor')
            if 'optimizer' in saved:
                states=saved['optimizer']['state']
                require(len(states)==304,'Missing optimizer states')
                require(all(int(v['step'])==saved['step'] for v in states.values()),'Optimizer step mismatch')
                require(all(k in saved for k in ['rng_local','rng_python','rng_torch','rng_cuda','extra']),'Missing resume RNG')
                require(saved['fingerprint']==fingerprint(config),'Checkpoint fingerprint mismatch')
            if name=='best.pt':require(saved['step']==selected['step'],'Best checkpoint differs from validation selection')
            snapshots[name]={'step':saved['step'],'tensors':len(saved['model']),'optimizer': 'optimizer' in saved}
        return {'history_through_step':max(steps),'validation_steps':expected,'best_step':selected['step'],
                'best_macro_r1':selected['validation']['macro_r1'],'checkpoints':snapshots}
    check('training_history_checkpoints_and_implementation',training)
    def selected_export():
        root=STORE/'validation_selected_step250'
        audit=read(root/'general_model.audit.json')
        require(audit['passed'] and audit['step']==250,'Step250 export audit missing')
        require(audit['checkpoint_sha256']==digest(root/'checkpoint.pt'),'Selected checkpoint bytes changed')
        require(audit['export_sha256']==digest(root/'general_model.pt'),'Selected export bytes changed')
        checkpoint=torch.load(root/'checkpoint.pt',weights_only=True,map_location='cpu')
        bundle=torch.load(root/'general_model.pt',weights_only=True,map_location='cpu')
        require(set(checkpoint['model'])==set(bundle['model']) and len(bundle['model'])==304,'Incomplete exported trained state')
        require(all(torch.equal(v,bundle['model'][n]) for n,v in checkpoint['model'].items()),'Export tensor mismatch')
        require(max(audit['embedding_max_abs_errors'])<=1e-6,'Export real-input equality failed')
        return {'step':250,'exact_trained_tensors':304,'real_input_audit_errors':audit['embedding_max_abs_errors']}
    check('selected_step250_export_all_tensors',selected_export)
    def frozen_queue():
        root=STORE/'lora_freeze_matched_v1'
        protocol=read(root/'evaluation_protocol.json')
        for path,sha in protocol['scripts'].items():require(digest(path)==sha,'Queued evaluator implementation changed: '+path)
        require(digest('experiments/lora_freeze_v5/pipeline.py')==protocol['pipeline_sha256'],'Queued pipeline changed')
        require(digest(root/'protocol.json')==protocol['training_protocol_sha256'],'Queued training protocol changed')
        status=read(root/'status.json')
        import psutil
        intentionally_stopped=status.get('state')=='stopped_before_training' and (root/'STOP').exists()
        require(status.get('state')=='completed' or intentionally_stopped or
                (isinstance(status.get('pid'),int) and psutil.pid_exists(status['pid'])),
                'Queued frozen-control worker absent without an intentional stop')
        return {'state':status['state'],'sealed_dependencies_unchanged':True,
                'not_yet_training':status['state'].startswith('waiting') or intentionally_stopped,
                'intentional_leakage_review_stop':intentionally_stopped}
    check('queued_frozen_lora_control_implementation',frozen_queue)
    def replay_validation(tag,initial_kind=None):
        # Match the scoring BLAS thread count of the original evaluator. Nearly
        # tied duplicate answers can choose different candidate IDs at 2 vs 4
        # threads, despite identical relevance and aggregate scores.
        torch.set_num_threads(4 if initial_kind else 2)
        result=read(RESULTS/'validation_controls'/f'{tag}.json')
        prov=result['provenance'];require(prov['protocol']==protocol,'Protocol differs')
        root=STORE/'initial_validation/cuda' if initial_kind else STORE/'validation_controls'/tag
        require(read(root/'provenance.json')==prov,'Cache/result provenance differs')
        require(read(root/'status.json')['state']=='completed','Incomplete extraction')
        def get(obj):
            sig=fingerprint({'object':obj,'provenance':prov})
            saved=torch.load(root/'vectors'/f'{sig}.pt',weights_only=True,map_location='cpu')
            require(saved['fingerprint']==sig,'Cache fingerprint differs')
            v=saved[initial_kind or 'vector']
            require(v.ndim==1 and torch.isfinite(v).all() and abs(float(v.norm())-1)<1e-4,'Invalid cached vector')
            return v
        q=torch.stack([get(r['q']) for r in items]);d=torch.stack([get(r['d']) for r in items]);t=torch.stack([get({'text':r['q']['text']}) for r in qa])
        for mode,subset,qv,dv in [('full',items,q,d),('qa_full',qa,q[qa_ids],d[qa_ids]),('qa_text_only',qa,t,d[qa_ids])]:
            check_predictions(subset,result[mode])
            replay=score_vectors(subset,qv,dv)
            require(replay['predictions']==result[mode]['predictions'],'Vector replay predictions differ: '+mode)
        torch.set_num_threads(2)
        return {'queries':len(items),'qa_queries':len(qa),'macro_r1':result['full']['macro_r1'],'all_cached_predictions_replayed':True,
                'scoring_threads':4 if initial_kind else 2}
    for tag,kind in [('raw_cuda','raw'),('step0_reconstructed_cuda','step0_reconstructed'),
                     ('general_3cad7dfc34c7_cuda',None),('qwen2b_cpu',None),('qwen_api_cpu',None)]:
        check('validation_'+tag,lambda tag=tag,kind=kind:replay_validation(tag,kind))
    def majority():
        result=read(RESULTS/'validation_controls/majority.json')
        require(result['protocol']==protocol,'Majority data differs')
        require(result['controls']==majority_controls(items,all_rows),'Majority recomputation differs')
        return {'train_only_recomputed':True}
    check('train_only_majority_controls',majority)
    def choice():
        from dataclasses import asdict
        from superembedding.heads import AttributeBasis
        from scripts.evaluate_choice_adaptation import class_metrics
        root=STORE/'choice_adaptation_cpu';r=read(RESULTS/'choice_adaptation_cpu.json');p=r['provenance'];design=p['design']
        require(p==read(root/'sealed_protocol.json'),'Choice seal differs')
        require(p['design_sha256']==digest('sources/choice_adaptation_v5.json'),'Choices changed after sealing')
        require(p['script_sha256']==digest(root/'implementation.py'),'Choice implementation snapshot differs')
        require(set(r['results'])=={'step250','step0_reconstructed','raw_pretrained'},'Missing model arm')
        count=0
        for label,sources in r['results'].items():
            require(set(sources)==set(design['sources']),'Missing choice source')
            for source,arms in sources.items():
                subset=[v for v in items if v['source']==source];original={}
                require(set(arms)==set(design['arms']),'Missing choice arm')
                for arm in design['arms']:
                    directions=design['directions'][arm]
                    if isinstance(directions,dict):directions=directions[source]
                    basis=AttributeBasis(tuple(directions),design['instruction'])
                    def get(obj):
                        model='step0_reconstructed' if label=='raw_pretrained' else label
                        sig=fingerprint({'object':obj,'basis':asdict(basis),'model':model,'provenance':p})
                        value=torch.load(root/'vectors'/f'{sig}.pt',weights_only=True,map_location='cpu')
                        require(value['fingerprint']==sig,'Choice cache mismatch')
                        return value['raw' if label=='raw_pretrained' else 'head']
                    q=torch.stack([get(v['q']) for v in subset]);d=torch.stack([get(v['d']) for v in subset]);t=torch.stack([get({'text':v['q']['text']}) for v in subset])
                    if arm=='original':original=d
                    require(set(arms[arm])=={'both_sides','query_only_original_index'},'Missing indexing arm')
                    for indexing,entry in arms[arm].items():
                        bank=d if indexing=='both_sides' else original
                        for mode,vectors in [('full',q),('text_only',t)]:
                            check_predictions(subset,entry[mode]);replay=score_vectors(subset,vectors,bank)
                            require(replay['predictions']==entry[mode]['predictions'],'Choice cache replay differs')
                            if source=='vsr':require(class_metrics(subset,replay)==entry[mode+'_class_metrics'],'VSR class metrics differ')
                            count+=1
        require(len(r['contrasts'])==24 and len(r['training_interactions'])==4,'Missing paired contrasts')
        return {'scored_cells':count,'models':3,'sources':2,'choices':3,'index_modes':2,'input_modes':2,'cached_replay':True}
    check('choice_all_72_cells_and_cache_replay',choice)
    def ablation():
        selection_path=STORE/'general_ablation/experiments/selection_sealed.json'
        r=read(RESULTS/'general_ablation_results.json');seal=read(selection_path)
        require(len(seal['results'])==36 and len(seal['selected'])==6 and len(r['runs'])==18,'Ablation grid incomplete')
        require(r['selection_sha256']==digest(selection_path),'Ablation seal changed')
        for run in seal['results']:
            path=Path(run['run']);best=torch.load(path/'best.pt',weights_only=True,map_location='cpu');last=torch.load(path/'last.pt',weights_only=True,map_location='cpu')
            require(best['step']==run['best_step'] and last['step']==1000,'Ablation checkpoint incomplete')
            require(best['fingerprint']==fingerprint(run['config']),'Ablation config differs')
            require(all(k in last for k in ['optimizer','rng_torch','rng_local']),'Ablation resume missing')
        return {'training_runs':36,'sealed_selected_evaluations':18,'checkpoints_verified':72}
    check('frozen_architecture_grid',ablation)
    def scienceqa(tag):
        import pyarrow.parquet as pq
        root=STORE/'mmeb_scienceqa';rows=pq.read_table(root/'test.parquet').to_pylist();texts=sorted({t for r in rows for t in r['tgt_text']});index={t:i for i,t in enumerate(texts)}
        result=read(RESULTS/f'scienceqa_{tag}.json');out=Path('runs/v5/scienceqa')/tag;prov=read(out/'provenance.json')
        require(prov==result['provenance'],'ScienceQA provenance differs')
        require(len(rows)==1000 and len(result['predictions'])==1000,'ScienceQA incomplete')
        def load(kind,objects):
            tensors=[]
            for start in range(0,len(objects),8):
                saved=torch.load(out/f'{kind}_{start:06d}.pt',weights_only=True,map_location='cpu')
                sig=fingerprint({'objects':objects[start:start+8],'provenance':prov})
                require(saved['fingerprint']==sig,'ScienceQA cached request differs')
                tensors.append(saved['vectors'])
                if kind=='target' and tag.endswith('_text_only'):
                    full=torch.load(out.parent/tag.removesuffix('_text_only')/f'target_{start:06d}.pt',weights_only=True,map_location='cpu')
                    require(torch.equal(full['vectors'],saved['vectors']),'Text-only candidate bank changed')
            return torch.cat(tensors)
        # Fingerprinted historical request path remains the C: junction.
        data=Path('C:/Users/123/.cache/superembedding/v5/mmeb_scienceqa')
        queries=[{'text':r['qry_text'].replace('<|image_1|>',''),**({} if tag.endswith('_text_only') else {'image':str(data/r['qry_img_path'])})} for r in rows]
        q=load('query',queries);d=load('target',[{'text':t} for t in texts]);correct=[]
        for i,row in enumerate(rows):
            names=row['tgt_text'];scores=q[i]@d[[index[t] for t in names]].T
            order=sorted(range(len(names)),key=lambda k:hashlib.sha256(names[k].encode()).hexdigest());selected=order[int(scores[order].argmax())]
            p={'query':i,'candidate_index':selected,'correct':names[selected]==names[0]}
            require(p==result['predictions'][i],'ScienceQA prediction replay differs');correct.append(p['correct'])
        require(abs(sum(correct)/1000-result['hit@1'])<1e-8,'ScienceQA metric differs')
        return {'queries':1000,'candidates':len(texts),'hit@1':result['hit@1'],'cached_replay':True}
    for tag in ['general_3cad7dfc34c7_cpu','general_3cad7dfc34c7_cpu_text_only']:
        check('scienceqa_'+tag,lambda tag=tag:scienceqa(tag))
    def transfer(tag):
        from superembedding.general import GENERAL_BASIS
        r=read(RESULTS/f'transfer_{tag}.json');cache=Path('runs/v5/transfer')/tag
        def get(obj):
            sig=fingerprint({'object':obj,'checkpoint':tag,'basis':list(GENERAL_BASIS.directions)})
            saved=torch.load(cache/(sig+'.pt'),weights_only=True,map_location='cpu')
            require(saved['fingerprint']==sig,'Transfer cache fingerprint differs')
            v=saved['vector'];require(torch.isfinite(v).all() and abs(float(v.norm())-1)<1e-4,'Invalid transfer vector');return v
        urban=Path('C:/Users/123/.cache/superembedding/v5/urban1k')
        rows=[x for x in read(urban/'annotations/annotations.json')['annotations'] if x['caption'].strip()]
        require(len(rows)==999,'Urban1k incomplete')
        q=torch.stack([get({'text':x['caption']}) for x in rows]);d=torch.stack([get({'images':[str(urban/'images'/x['image'])]}) for x in rows]);scores=d@q.T
        for direction,s in [('image_to_text',scores),('text_to_image',scores.T)]:
            ranks=s.argsort(-1,descending=True)
            for k in [1,5,10]:
                val=(ranks[:,:k]==torch.arange(999)[:,None]).any(-1).float().mean().item()
                require(abs(val-r['urban1k_common_999'][f'{direction}_R@{k}'])<1e-7,'Urban score replay differs')
        root=Path('data/evidence_diagnostic_v1');cases=[json.loads(s) for s in (root/'records.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(cases)==len(r['cases'])==256,'Missing evidence cases')
        for case,reported in zip(cases,r['cases']):
            require(case['id']==reported['id'],'Evidence order changed')
            a,b,c=[get({'images':[str(root/case[k])]}) for k in ['anchor','relevant_change','irrelevant_change']];q=get({'text':case['query']})
            for field,value in [('anchor_margin',float(q@a)-float(q@b)),('nuisance_margin',float(q@c)-float(q@b)),
                                ('target_change_distance',float(1-a@b)),('nuisance_distance',float(1-a@c))]:
                require(abs(value-reported[field])<1e-7,'Evidence margin replay differs')
        return {'urban_pairs':999,'evidence_cases':256,'scene_groups':64,'cached_replay':True,
                'limitation':'Historical transfer cache does not bind image bytes or encoder code; replay validates current artifact consistency, not historical immutability.'}
    for tag in ['general_3cad7dfc34c7_cpu','omnijev_raw_cpu']:
        check('transfer_'+tag,lambda tag=tag:transfer(tag))
    # Pending is distinct from failed, and is never silently treated as complete.
    pending.extend([
        'LEAKAGE BLOCK: CUB/DTD confirmed duplicate photos/scenes; V5 DocVQA entity overlap; all historical results exploratory until versioned remediation and fresh runs',
        '2000-step general training and final validation-selected export/evaluation',
        'Matched frozen-versus-LoRA run waits for parent training AND final evaluation',
        'VLM2Vec/GME/WeMM/Ovis inference comparisons (download is not evaluation)',
        'Unseen natural/compositional evidence and broader data/method study',
        'True single-branch/direction-count/decision-pretraining controls and general LoRA seed replication',
        'Final data rights/model card/release audit; no Git publication until complete'])
    missing=[]
    for name in ['scienceqa_omnijev_raw_cpu.json','scienceqa_omnijev_raw_cpu_text_only.json']:
        if not (RESULTS/name).exists():missing.append(name)
    if missing:pending.append({'same_precision_external_raw_controls_missing':missing})
    else:
        for tag in ['omnijev_raw_cpu','omnijev_raw_cpu_text_only']:
            check('scienceqa_'+tag,lambda tag=tag:scienceqa(tag))
    from superembedding.leakage_guard import promotion_problems
    leakage_blockers=promotion_problems()
    report={'utc':datetime.now(timezone.utc).isoformat(),'passed_checks':checks,'failed_checks':failures,
            'leakage_clearance_passed':not leakage_blockers,'leakage_blockers':leakage_blockers,
            'pending':pending,'current_artifacts_integrity_passed':not failures,'whole_project_complete':False,
            'audit_scope':'Read-only full cached prediction replay for selected tasks, checkpoint integrity and arm coverage; not fresh model inference or a claim of research completeness.'}
    status_path=STORE/'external_control_completion/status.json'
    if status_path.exists():report['external_control_completion']=read(status_path)
    input_audit=STORE/'external_input_audit/external_inputs_retrospective_audit.json'
    if input_audit.exists():report['retrospective_input_inventory']=read(input_audit)
    atomic_json(report,RESULTS/'experiment_completeness_audit.json')
    trained=next((x['detail'] for x in checks if x['check']=='training_history_checkpoints_and_implementation'),{})
    lines=['# 当前实验完整性审计','',f"审计时间：{report['utc']}。完整项目完成：否。",'',
           '数据泄露审查未通过：CUB/DTD 已确认跨划分同图/同场景；V5 DocVQA 存在实体和固定答案交叉。下列数值回放通过不代表实验无泄露。详见 DATA_LEAKAGE_REVIEW.zh.md。','',
           f"主训练已审计到 Step {trained.get('history_through_step')}，验证最佳为 Step {trained.get('best_step')}，8 来源平均 R@1 为 {trained.get('best_macro_r1',0)*100:.2f}%。实时步数以训练状态文件为准。",'',
           '| 检查 | 状态 |','|---|---|']
    lines += [f"| {r['check']} | 通过 |" for r in checks]
    lines += [f"| {r['check']} | 失败：{r['message']} |" for r in failures]
    lines += ['','本次从实际缓存向量重算检索预测和分数，验证分组覆盖与断点；不把运行状态 completed 当作充分证据。','',
              '## 尚未完成','']+[f'- {r}' for r in pending]
    lines += ['','## 已修复与证据边界','',
              '历史 transfer 缓存键缺少输入字节与实现哈希，当前回放一致不代表历史文件从未变化。已建立事后输入清单；最终训练评测入口新增评测前/后输入及代码哈希校验，不改动其他队列已封存的评测脚本。',
              '', 'ScienceQA 的 CPU 同精度 raw 与 text-only 缺项由独立后台队列补跑，状态在 external_control_completion/status.json；仅当结果存在并通过回放才算完成。',
              '', '冻结/LoRA 测试加入默认 pytest；同时修复其导入时全局更改 CUDA_VISIBLE_DEVICES 的副作用。合并后 70 项测试通过，包括 CPU/CUDA 精确断点回放。该修复只涉及测试，不修改现有训练实现。',
              '', '初次数值审计发现 BLAS 使用 2 与 4 个 CPU 线程时，raw/Step 0 的重复答案可能选到不同的候选记录 ID，但正确性和全部分数一致。按原执行器线程数重算后，所有预测记录逐条一致；保留线程设置作为复现条件。',
              '', '## 结果不能被解释为全面成功','',
              'Step 250 专精 choice：ChartQA 双方重编码从 16.92% 降至 7.69%；VSR 总分从 55.45% 升至 56.36%，但 balanced accuracy 从 59.42% 降至 57.88%，text-only 反而更高。这不支持当前提示方案改善视觉泛化。',
              '', 'Step 250 外部 ScienceQA 为 14.4%，text-only 7.1%；合成证据宏平均从 raw 的 26.17% 升至 51.95%，但空间项差值区间跨零。更强基线和自然证据、最终模型结果仍需完整报告。', '',
              '不修改已封存训练或评测实现，不按外部结果选模型；不提交 Git。','']
    Path('reports/EXPERIMENT_COMPLETENESS_V5.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
