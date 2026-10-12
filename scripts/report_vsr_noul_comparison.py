"""Replay completed VSR caches; no inference, training, threshold fit or selection."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from experiments.vsr_decision_v5 import run as ref
from superembedding.heads import OptionEmbeddingHead
from superembedding.checkpoint import fingerprint, atomic_json


def calibrated_noul(mu, meta):
    p=float(mu[0]); bias=float(meta.get('biases',{}).get('noul',0))
    if bias:
        p=min(max(p,1e-9),1-1e-9)
        z=max(-40.,min(40.,math.log(p/(1-p))+bias))
        p=1/(1+math.exp(-z))
    temp=float(meta.get('temperatures',{}).get('noul',meta.get('temperature',1)))
    if abs(temp-1)>=1e-6:
        a=max(1e-9,p)**(1/temp);b=max(1e-9,1-p)**(1/temp);p=a/(a+b)
    return round(p,4)


def main():
    torch.set_num_threads(1)
    items,protocol=ref.prepare();sig=ref.signature(protocol)
    labels={r['id']:r['key'] for r in items}; arms={}; inputs={}; checks=[]
    meta=ref.read(ref.ROOT/'models/OmniJev-0.8B-v1.1/head_meta.json')
    for arm in ['original','step500']:
        path=ref.OUT/arm/'result.json';saved=ref.read(path)
        if saved['protocol_hash']!=sig:raise ValueError('Result protocol changed')
        inputs[str(path)]=ref.sha(path)
        audit=ref.read(ref.OUT/arm/'audit.json')
        assert audit['head_original_exact'] and all(v['official_equal'] for v in audit['native_api'])
        assert (audit['lora_before']==audit['lora_after'])==(arm=='original')
        for typ in ['choice','noul']:
            for mode in ['full','text_only']:
                result=saved['results'][typ+'_'+mode];preds=result['predictions']
                assert [p['id'] for p in preds]==[r['id'] for r in items]
                for p in preds:
                    assert p['actual']==labels[p['id']]
                    key=ref.signature({'id':p['id'],'mode':mode,'type':typ,'protocol':sig,'arm':arm})
                    cached=ref.read(ref.OUT/arm/'cache'/(key+'.json'))
                    assert cached['key']==key and cached['prediction']==p
                    if typ=='noul':
                        prob=calibrated_noul(p['raw_mu'],meta)
                        assert prob==p['answer']['noul']
                        assert p['predicted']==('yes' if prob>=.5 else 'no')
                    else:
                        assert p['predicted']==p['answer']['choice']
                assert ref.metrics(preds)==result['metrics']
                arms[arm+'/'+typ+'/'+mode]=result
        checks.append(arm+': 440 decision predictions matched per-query caches; 220 noul calibrations/thresholds replayed')
        if arm=='step500':
            state=torch.load(ref.STEP500,map_location='cpu',weights_only=True)
            assert state['step']==500
            head=OptionEmbeddingHead(1024,embed_dim=256,mode='raw').eval()
            head.load_state_dict({k.removeprefix('embedding_head.'):v for k,v in state['model'].items() if k.startswith('embedding_head.')},strict=True)
            def vector(obj):
                key=ref.signature({'object':obj,'protocol':sig,'arm':arm})
                cached=torch.load(ref.OUT/arm/'cache'/(key+'.pt'),map_location='cpu',weights_only=True)
                assert cached['key']==key and cached['u'].shape==(4,1024)
                assert torch.isfinite(cached['u']).all()
                with torch.no_grad():return head(cached['u']).embedding
            targets={s:vector({'text':s}) for s in {r['d']['text'] for r in items}}
            for mode in ['full','text_only']:
                queries={r['id']:vector(r['q'] if mode=='full' else {'text':r['q']['text']}) for r in items}
                preds=ref.embedding_predictions(items,queries,targets)
                result=saved['results']['embedding_'+mode]
                assert preds==result['predictions'] and ref.metrics(preds)==result['metrics']
                arms['step500/embedding/'+mode]=result
            checks.append('Step500: full/text-only projection and ranking replayed from cached hidden states and immutable checkpoint')
    for model in ['qwen2b','qwen_api']:
        path=ref.ROOT/'artifacts/results/v5/validation_controls'/(model+'_cpu.json')
        data=ref.read(path);prov=data['provenance'];inputs[str(path)]=ref.sha(path)
        assert prov['protocol']==protocol['validation_protocol']
        cache=ref.STORE/'validation_controls'/(model+'_cpu')
        assert ref.read(cache/'provenance.json')==prov
        def cached_vector(obj):
            key=fingerprint({'object':obj,'provenance':prov})
            row=torch.load(cache/'vectors'/(key+'.pt'),map_location='cpu',weights_only=True)
            assert row['fingerprint']==key and torch.isfinite(row['vector']).all()
            return row['vector']
        # Match original evaluator scoring thread count, including duplicate-answer ties.
        torch.set_num_threads(2)
        targets={s:cached_vector({'text':s}) for s in {r['d']['text'] for r in items}}
        for mode,field in [('full','full'),('text_only','qa_text_only')]:
            queries={r['id']:cached_vector(r['q'] if mode=='full' else {'text':r['q']['text']}) for r in items}
            preds=ref.embedding_predictions(items,queries,targets)
            reported=[p for p in data[field]['predictions'] if p['id'] in labels]
            assert len(reported)==110
            for p,old in zip(preds,reported):
                assert p['id']==old['id'] and p['candidate_id']==old['candidate_id']
                assert (p['predicted']==p['actual'])==old['correct']
            arms[model+'/'+mode]={'metrics':ref.metrics(preds),'predictions':preds}
        checks.append(model+': full/text-only vectors, candidate IDs and correctness replayed')
    baseline=[{'id':r['id'],'actual':r['key'],'predicted':'yes'} for r in items]
    arms['always_yes']={'metrics':ref.metrics(baseline),'predictions':baseline}
    pairs=[('step500/embedding/full','original/noul/full'),('step500/embedding/full','qwen2b/full'),
           ('step500/embedding/full','qwen_api/full'),('step500/noul/full','original/noul/full')]
    pairs += [(k,k.removesuffix('/full')+'/text_only') for k in arms if k.endswith('/full')]
    comparisons={a+' minus '+b:ref.paired(arms[a]['predictions'],arms[b]['predictions']) for a,b in pairs}
    overlap=ref.read(ref.ROOT/'artifacts/results/v5/question_overlap_audit.json')
    seen={p['id'] for p in overlap['predictions'] if p['source']=='vsr' and p['seen_question']}
    subsets={}
    for key,result in arms.items():
        subsets[key]={}
        for name,ids in [('seen_exact_question',seen),('unseen_exact_question',set(labels)-seen)]:
            preds=[p for p in result['predictions'] if p['id'] in ids]
            subsets[key][name]=ref.metrics(preds)
    result={'requested_comparison_complete':True,'extended_frozen_projection_comparison_complete':False,
            'protocol_sha256':ref.sha(ref.OUT/'protocol.json'),'input_result_sha256':inputs,
            'audit_checks':checks,'arms':arms,'paired_comparisons':comparisons,'question_overlap_subgroups':subsets,
            'limits':['110 internal validation questions already observed, not untouched official VSR test.',
                      'Noul/embedding retain native different prompts; method comparison, not a pure head ablation.',
                      'Qwen2B is local Qwen3-VL-Embedding-2B; API is a mutable provider alias with unknown precision/size.',
                      'CPU FP32 OmniJev/Step500 values must not be substituted for CUDA BF16 scores.',
                      'No thresholds/prompts fit on these questions; prior validation exposure and pretraining contamination remain.',
                      'Cached replay verifies artifact consistency, not fresh backbone inference or absence of leakage.']}
    atomic_json(result,ref.ROOT/'artifacts/results/v5/vsr_noul_direct_comparison.json')
    order=['original/noul','step500/embedding','qwen2b','qwen_api','step500/noul','original/choice','step500/choice']
    lines=['# VSR 110 题：原生 noul、Qwen 与 Step 500','',
           '所请求对照已完成并重放缓存核验。Yes 68 / No 42；CPU FP32（API 精度由服务方管理）。这是事后内部验证诊断，不是无泄露主结果。','',
           '| 模型/接口 | 正确题数 | Accuracy | Balanced accuracy | Text-only accuracy |',
           '|---|---:|---:|---:|---:|']
    for key in order:
        m=arms[key+'/full']['metrics'];t=arms[key+'/text_only']['metrics']
        lines.append(f"| {key} | {m['correct']}/110 | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {t['accuracy']:.2%} |")
    lines+=['','Always-Yes：68/110，Accuracy 61.82%，Balanced accuracy 50%。Qwen2B 指本地 embedding 模型，不是生成式问答；Qwen API 独立列示，不能冒称某个公开参数规模。','',
            'noul 使用官方原 decision head、LM features、温度/偏置与四位小数概率，固定 P(yes)≥0.5，不在此数据上调阈值。两条原生 API 等价性检查已保存；本次逐题核验概率后处理。Step500 embedding 使用已训练 LoRA 和同一步 projection，逐题重放全部 110 个候选的检索。','',
            '| 配对比较 | Accuracy 差值（百分点） | 95% 分层配对 bootstrap 区间 |','|---|---:|---|']
    for a,b in pairs[:4]:
        r=comparisons[a+' minus '+b];lo,hi=r['accuracy_paired_bootstrap_95']
        lines.append(f"| {a} − {b} | {100*r['accuracy_delta']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
    lines+=['','区间仅描述这组查询的重采样不确定性，不含训练种子、方法选择或预训练污染，不作多重比较校正。','',
            '21/110 个完整问题（含指令、规范化后）在历史训练中出现过；JSON 另给 seen/unseen 分组，仅作诊断。换措辞不是新实体。VSR 旧验证也曾参与整体 checkpoint 选择，本次固定 Step500 不会使其变成未见测试。','',
            '原始流程另有冻结 projection 三个学习率对照，因父训练队列停止尚未完成；本报告不把缺失组算作已完成，也不为它们等待或修改正在续训的 GPU 任务。','',
            '机器可读结果与逐题预测：`artifacts/results/v5/vsr_noul_direct_comparison.json`。']
    (ref.ROOT/'reports/VSR_NOUL_DIRECT_COMPARISON.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'checks':checks,'metrics':{k:v['metrics'] for k,v in arms.items() if k.endswith('/full')},
                      'main_comparisons':{k:v for k,v in comparisons.items() if k in [a+' minus '+b for a,b in pairs[:4]]}},indent=2))


if __name__=='__main__':main()
