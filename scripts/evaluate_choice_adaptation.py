"""Frozen choice-text intervention; all arms sealed before any evaluation."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def class_metrics(items, result):
    from collections import Counter
    keys={r['id']:r['key'] for r in items}
    actual=Counter(keys.values());predicted=Counter();correct=Counter();confusion=Counter()
    for p in result['predictions']:
        a=keys[p['id']];b=keys[p['candidate_id']]
        predicted[b]+=1;correct[a]+=int(a==b);confusion[(a,b)]+=1
    return {'balanced_accuracy':sum(correct[k]/n for k,n in actual.items())/len(actual),
            'per_class_recall':{k:correct[k]/n for k,n in actual.items()},
            'predicted_class_counts':dict(predicted),
            'confusion':[{'actual':a,'predicted':b,'count':n} for (a,b),n in sorted(confusion.items())]}


def paired_stats(values, seed=20261011):
    import numpy as np
    v=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    boots=v[rng.integers(0,len(v),size=(5000,len(v)))].mean(-1)
    return {'delta':float(v.mean()),'paired_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
            'queries':len(v),'resampling':'5000 paired query resamples; exploratory unadjusted interval'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--seal-only',action='store_true');ap.add_argument('--wait-qwen',action='store_true')
    args=ap.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['MSO_FLA']='0';os.environ['MSO_BRANCH_CHUNK']='4';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch.nn import functional as F
    from filelock import FileLock
    from superembedding.checkpoint import atomic_json,atomic_save,fingerprint
    from superembedding.general_export import load_bundle_encoder
    from superembedding.general import GENERAL_BASIS
    from superembedding.heads import AttributeBasis
    from superembedding.validation_protocol import load_protocol,score_vectors,majority_controls
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    store=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    out=store/('choice_adaptation_'+args.device);out.mkdir(exist_ok=True)
    design_path=Path('sources/choice_adaptation_v5.json');design=json.loads(design_path.read_text(encoding='utf-8'))
    protocol,all_items,all_rows=load_protocol(store/'data_v51')
    items=[r for r in all_items if r['source'] in design['sources']]
    if len(items)!=175:raise ValueError('Unexpected coverage')
    checkpoints={'step250':store/'validation_selected_step250/general_model.pt',
                 'step0_reconstructed':store/f'step0_reconstructed_{args.device}/general_model.pt'}
    prov={'design':design,'design_sha256':hashlib.sha256(design_path.read_bytes()).hexdigest(),
          'validation_protocol':protocol,'ids':[r['id'] for r in items],
          'checkpoint_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in checkpoints.items()},
          'device':args.device,'dtype':'float32' if args.device=='cpu' else 'bfloat16',
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with FileLock(str(out/'evaluation.lock'),timeout=0):
        seal=out/'sealed_protocol.json'
        if seal.exists() and json.loads(seal.read_text(encoding='utf-8'))!=prov:raise ValueError('Sealed protocol changed')
        if not seal.exists():atomic_json(prov,seal)
        (out/'implementation.py').write_bytes(Path(__file__).read_bytes())
        if args.seal_only:
            print(json.dumps({'sealed':str(seal),'queries':len(items)}),flush=True);return
        if args.wait_qwen:
            deadline=time.monotonic()+8*3600
            while True:
                path=store/'validation_controls/qwen2b_cpu/status.json'
                status=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if status.get('state')=='completed':break
                if time.monotonic()>deadline:raise RuntimeError('Qwen prerequisite wait expired')
                atomic_json({'state':'waiting_for_qwen2b','pid':os.getpid(),'reason':'Avoid simultaneous loading of both CPU model backbones'},out/'status.json')
                time.sleep(15)
        for row in items:
            if hashlib.sha256(Path(row['q']['images'][0]).read_bytes()).hexdigest()!=row['image_file_sha256']:
                raise ValueError('Input image changed')
        if design['directions']['original']!=list(GENERAL_BASIS.directions) or design['instruction']!=GENERAL_BASIS.instruction:
            raise ValueError('Original basis is not training basis')
        result={};started=time.perf_counter()
        for model_name,checkpoint in checkpoints.items():
            model=None;computed=0
            for source in design['sources']:
                rows=[r for r in items if r['source']==source]
                target_banks={}
                for arm in design['arms']:
                    directions=design['directions'][arm]
                    if isinstance(directions,dict):directions=directions[source]
                    basis=AttributeBasis(tuple(directions),design['instruction'])
                    if len(basis.directions)!=4:raise ValueError('Slot count changed')
                    def encode(obj):
                        nonlocal model,computed
                        sig=fingerprint({'object':obj,'basis':asdict(basis),'model':model_name,'provenance':prov})
                        path=out/'vectors'/f'{sig}.pt'
                        if path.exists():
                            saved=torch.load(path,weights_only=True,map_location='cpu')
                            if saved['fingerprint']!=sig:raise ValueError('Cached vector mismatch')
                        else:
                            if model is None:
                                model,bundle=load_bundle_encoder(checkpoint)
                                if bundle['step']!=(250 if model_name=='step250' else 0):raise ValueError('Wrong frozen checkpoint')
                                model.requires_grad_(False)
                            # Validate the original bundle first; apply only this
                            # explicitly recorded inference-time prompt intervention.
                            model.basis=basis
                            with torch.no_grad():
                                u,_=model.extract_features(obj)
                                saved={'head':model.embedding_head(u).embedding.cpu(),
                                       'raw':F.normalize(F.normalize(u.float(),dim=-1).flatten(),dim=-1).cpu(),
                                       'fingerprint':sig}
                            atomic_save(saved,path);computed+=1
                            if computed%25==0 or computed==1:
                                state={'state':'encoding','model':model_name,'source':source,'arm':arm,'new_objects_for_model':computed,
                                       'pid':os.getpid(),'seconds':time.perf_counter()-started}
                                atomic_json(state,out/'status.json');print(json.dumps(state),flush=True)
                        for kind in ['head','raw']:
                            v=saved[kind]
                            if v.ndim!=1 or not torch.isfinite(v).all() or abs(float(v.norm())-1)>1e-4:
                                raise ValueError('Invalid vector')
                        return saved
                    q=[encode(r['q']) for r in rows];d=[encode(r['d']) for r in rows]
                    qt=[encode({'text':r['q']['text']}) for r in rows]
                    variants=[('head',model_name)]
                    if model_name=='step0_reconstructed':variants.append(('raw','raw_pretrained'))
                    for kind,label in variants:
                        qv=torch.stack([r[kind] for r in q]);dv=torch.stack([r[kind] for r in d]);tv=torch.stack([r[kind] for r in qt])
                        if arm=='original':target_banks[kind]=dv
                        for indexing,bank in [('both_sides',dv),('query_only_original_index',target_banks[kind])]:
                            full=score_vectors(rows,qv,bank);text=score_vectors(rows,tv,bank)
                            entry={'full':full,'text_only':text}
                            if source=='vsr':
                                entry['full_class_metrics']=class_metrics(rows,full)
                                entry['text_only_class_metrics']=class_metrics(rows,text)
                            result.setdefault(label,{}).setdefault(source,{}).setdefault(arm,{})[indexing]=entry
                atomic_json({'state':'partial','provenance':prov,'results':result},out/'partial_results.json')
            if model is not None:del model
            gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
        contrasts=[]
        def hits(entry,mode):return [float(p['correct']) for p in entry[mode]['predictions']]
        for label,sources in result.items():
            for source,arms in sources.items():
                for arm in ['paraphrase','specialized']:
                    for indexing in ['both_sides','query_only_original_index']:
                        base=arms['original'][indexing];changed=arms[arm][indexing]
                        a,b=hits(changed,'full'),hits(base,'full')
                        at,bt=hits(changed,'text_only'),hits(base,'text_only')
                        contrasts.append({'model':label,'source':source,'arm':arm,'indexing':indexing,
                                          'full_improvement':paired_stats([x-y for x,y in zip(a,b)]),
                                          'change_in_visual_contribution':paired_stats([x-y-u+v for x,y,u,v in zip(a,at,b,bt)])})
        training_interactions=[]
        for source in design['sources']:
            for indexing in ['both_sides','query_only_original_index']:
                deltas=[]
                for label in ['step250','step0_reconstructed']:
                    a=result[label][source]
                    deltas.append([x-y for x,y in zip(hits(a['specialized'][indexing],'full'),hits(a['original'][indexing],'full'))])
                training_interactions.append({'source':source,'indexing':indexing,
                    'prompt_benefit_step250_minus_step0':paired_stats([a-b for a,b in zip(*deltas)])})
        final={'provenance':prov,'results':result,'contrasts':contrasts,'training_interactions':training_interactions,
               'majority':majority_controls(items,all_rows),'test_based_prompt_selection':False,
               'interpretation':design['inference'],'limitations':design['claim_boundary']}
        destination=Path('artifacts/results/v5')/f'choice_adaptation_{args.device}.json'
        atomic_json(final,destination)
        lines=['# 冻结 Step 250 的 choice 专精实验','',design['claim_boundary'],'',
               '所有方向在评测前封存；全部报告，没有按结果选择最佳提示。保持四方向、共享投影和输入文字，只有 choice 文本改变。CPU FP32 与原 GPU 验证分数分开；组内比较精度一致。','',
               '| 模型 | 来源 | 方向 | 索引 | R@1 % | text-only % | VSR balanced accuracy % |','|---|---|---|---|---:|---:|---:|']
        for label,sources in result.items():
            for source,arms in sources.items():
                for arm,indices in arms.items():
                    for indexing,r in indices.items():
                        balanced=f"{r['full_class_metrics']['balanced_accuracy']*100:.2f}" if source=='vsr' else '—'
                        lines.append(f"| {label} | {source} | {arm} | {indexing} | {r['full']['macro_r1']*100:.2f} | {r['text_only']['macro_r1']*100:.2f} | {balanced} |")
        lines+=['','both_sides 对查询和候选都用任务固定方向，须重建该任务索引；query_only_original_index 只改查询方向并保留原候选向量。没有按每个查询生成候选向量。','',
                'JSON 包含问题类型、VSR 混淆矩阵/预测类别分布、相对原方向的成对区间、视觉贡献变化及训练前后提示收益差。区间为探索性，未校正多重检验。小样本和已见训练来源不能支持未见任务泛化。','']
        Path(f'reports/CHOICE_ADAPTATION_{args.device.upper()}_V5.zh.md').write_text('\n'.join(lines),encoding='utf-8')
        atomic_json({'state':'completed','result':str(destination),'seconds':time.perf_counter()-started},out/'status.json')
        print(json.dumps({'completed':str(destination)}),flush=True)


if __name__=='__main__':
    main()
