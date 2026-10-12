"""Isolated CPU VSR decision/embedding comparison; never changes a training run."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STORE = Path('D:/DEMO/superembedding-storage/superembedding/v5')
OUT = STORE/'vsr_decision_step500_v1'
FROZEN = STORE/'lora_freeze_matched_v1'
STEP500 = STORE/'general_stage1_v51/step_000500_weights.pt'
INITIAL = STORE/'step0_reconstructed_cuda/checkpoint.pt'
LRS = ('1e-05', '0.0001', '0.001')
LABELS = ('yes', 'no')
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def cpu_setup():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MSO_FLA'] = '0'
    os.environ['MSO_BRANCH_CHUNK'] = '4'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Cannot set below-normal worker priority')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261011)


def items_and_protocol():
    from superembedding.validation_protocol import load_protocol
    protocol, items, _ = load_protocol(STORE/'data_v51')
    items = [r for r in items if r['source'] == 'vsr']
    if len(items) != 110 or Counter(r['key'] for r in items) != {'yes':68, 'no':42}:
        raise ValueError('VSR subset or labels changed')
    for r in items:
        if sha(r['q']['images'][0]) != r['image_file_sha256']:
            raise ValueError('VSR image changed')
    return items, protocol


def prepare():
    items, validation_protocol = items_and_protocol()
    files = [__file__, str(Path(__file__).with_name('test_run.py')),
             'superembedding/general.py', 'superembedding/heads.py', 'superembedding/omnijev.py',
             'superembedding/general_inference.py', 'superembedding/validation_protocol.py',
             'repos/OmniJev/mso/infer.py', 'repos/OmniJev/mso/records.py',
             'repos/OmniJev/mso/branch.py', 'repos/OmniJev/mso/head.py', 'repos/OmniJev/mso/v04.py',
             'models/OmniJev-0.8B-v1.1/head.pt', 'models/OmniJev-0.8B-v1.1/head_meta.json',
             'models/OmniJev-0.8B-v1.1/new_tok_emb.pt',
             'models/OmniJev-0.8B-v1.1/adapter_model.safetensors',
             'models/Qwen3.5-0.8B/source_revision.json']
    protocol = {'name':'vsr-native-decision-and-embedding-step500-v1',
        'validation_protocol':validation_protocol, 'items_sha256':signature(items),
        'implementation_and_sources':{str(f):sha(ROOT/f) for f in files},
        'initial_sha256':sha(INITIAL), 'step500_sha256':sha(STEP500),
        'frozen_protocol_sha256':sha(FROZEN/'protocol.json'),
        'device':'CPU FP32, one thread, below-normal priority', 'max_pixels':65536,
        'arms':['original decision choice', 'original decision noul supplementary',
                'step500 LoRA + original decision choice', 'step500 LoRA + original decision noul supplementary',
                'step500 LoRA + embedding', 'frozen LoRA + step500 projection for 3 predefined LRs'],
        'primary_frozen_lr':'1e-05', 'frozen_lrs':list(LRS),
        'modes':['full','text_only'], 'primary_choice':['Yes.','No.'],
        'decision_prompt':'Original sampled question; official choice/noul rendering, official LM features and calibration.',
        'embedding_prompt':'Existing training prefix and fixed four directions, all 110 target entries; unchanged.',
        'noul_rule':'Official calibrated rounded P(yes) >= 0.5; supplementary, not selected by accuracy.',
        'choice_rule':'Official choice argmax over Yes/No; record abstain probability, no abstention filtering.',
        'tie_rule':'Official first choice; embedding first candidate in record-id order.',
        'text_only':'Remove actual image tokens and pixels, no blank-image substitution.',
        'native_audit':'First two validation rows: custom full path must match official ask; swapped choices must agree.',
        'metrics':['accuracy','balanced_accuracy','yes/no recall','confusion','predicted_yes_rate',
                   'paired full-minus-text-only and model differences with bootstrap intervals'],
        'limitations':['Internal held validation subset, not official VSR test; native decision and embedding use their own prompts, not just a head swap.',
                      'Fixed step500 retrospective comparison, one seed; VSR not used to select an LR.',
                      'CPU scores can differ from previously reported CUDA BF16 scores.',
                      'Query-level bootstrap, not training-seed or pretraining-contamination uncertainty.']}
    path = OUT/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Sealed experiment changed; use a new output directory')
    if not path.exists():
        write(protocol, path)
    return items, protocol


def metrics(predictions):
    if not predictions or len({p['id'] for p in predictions}) != len(predictions):
        raise ValueError('Empty or duplicate predictions')
    c = Counter((p['actual'],p['predicted']) for p in predictions)
    if any(a not in LABELS or b not in LABELS for a,b in c):
        raise ValueError('Unknown label')
    counts = {a:sum(c[a,b] for b in LABELS) for a in LABELS}
    if min(counts.values()) == 0:
        raise ValueError('Both classes required')
    recalls = {a:c[a,a]/counts[a] for a in LABELS}
    n = len(predictions)
    return {'n':n, 'correct':sum(c[a,a] for a in LABELS),
            'accuracy':sum(c[a,a] for a in LABELS)/n,
            'balanced_accuracy':sum(recalls.values())/2, 'per_class_recall':recalls,
            'predicted_yes_rate':sum(c[a,'yes'] for a in LABELS)/n,
            'confusion':{a:{b:c[a,b] for b in LABELS} for a in LABELS}}


def paired(a, b):
    import numpy as np
    aa = {p['id']:p for p in a}; bb = {p['id']:p for p in b}
    if aa.keys() != bb.keys() or len(aa)!=len(a) or len(bb)!=len(b):
        raise ValueError('Paired prediction IDs differ')
    rng = np.random.default_rng(20261011)
    diffs, boots, weights = [], [], []
    for label in LABELS:
        ids = [i for i,p in aa.items() if p['actual']==label]
        if not ids or any(aa[i]['actual']!=bb[i]['actual'] for i in ids):
            raise ValueError('Paired labels differ')
        d = np.array([float(aa[i]['predicted']==label)-float(bb[i]['predicted']==label) for i in ids])
        diffs.append(d.mean()); weights.append(len(ids)/len(aa))
        boots.append(d[rng.integers(0,len(d),(5000,len(d)))].mean(-1))
    acc = np.average(boots, axis=0, weights=weights); ba = np.mean(boots,axis=0)
    return {'accuracy_delta':float(np.average(diffs,weights=weights)),
            'balanced_accuracy_delta':float(np.mean(diffs)),
            'accuracy_paired_bootstrap_95':np.quantile(acc,[.025,.975]).tolist(),
            'balanced_accuracy_paired_bootstrap_95':np.quantile(ba,[.025,.975]).tolist(),
            'unit':'5000 paired query resamples stratified by true label; unadjusted exploratory intervals'}


def tensor_sha(state):
    h = hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def native_question(item, qtype, reverse=False):
    prefix = 'Find the answer to this question: '
    if not item['q']['text'].startswith(prefix):
        raise ValueError('Unexpected embedding query prefix')
    q = {'type':qtype, 'instructions':item['q']['text'][len(prefix):]}
    if qtype=='choice':
        options = [{'key':'yes','text':'Yes.'},{'key':'no','text':'No.'}]
        q['options'] = list(reversed(options)) if reverse else options
    return q


def decision(mso, item, mode, qtype, reverse=False):
    """Official single-question branch path, extended only to allow zero images."""
    import torch
    from PIL import Image
    from mso import branch as BR, records as T
    q = native_question(item,qtype,reverse)
    keys,opts = mso._options(q)
    frames=[]
    if mode=='full':
        with Image.open(item['q']['images'][0]) as im:
            frames.append(im.convert('RGB'))
    content=[{'type':'image'} for _ in frames]+[{'type':'text','text':q['instructions']}]
    prompt=mso.proc.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True)
    prompt+=''.join(T.OPT_OPEN+T.render_option(o)+T.OPT_CLOSE for o in opts)
    enc=mso.proc(text=[prompt],images=frames or None,return_tensors='pt')
    if mode=='text_only' and any(k.startswith('pixel_values') for k in enc):
        raise ValueError('Image retained in text-only control')
    enc={k:v.to(mso.dev) for k,v in enc.items()}
    ids=enc['input_ids'][0].tolist(); opens,closes=mso._spans_of(ids)
    if len(opens)!=len(opts) or len(closes)!=len(opts):
        raise ValueError('Decision option tokenization mismatch')
    prefix=max(1,opens[0]-1)
    penc=BR.prefix_inputs(enc,prefix)
    positions=BR.prefix_positions(mso.model,penc)
    rows=BR.rows_from_ids(ids,prefix,opens,closes,k=len(opts))
    with torch.no_grad():
        [(u,zq,feats)]=BR.branch_questions(mso.model.backbone,penc,[rows],mso.dev,
            mso.open_ids,mso.close_ids,lm_head=mso.lm,prefix_pos=positions,pad_id=mso.pad_id)
        if mso.lm_feats and feats is None:
            raise ValueError('Official decision LM features omitted')
        mu=mso.model.head(u,zq,mso.type_id[qtype],feats)
    if not torch.isfinite(mu).all(): raise ValueError('Nonfinite decision scores')
    answer=mso._finish(q,keys,opts,mu,0.)
    pred=answer['choice'] if qtype=='choice' else ('yes' if answer['noul']>=.5 else 'no')
    return {'id':item['id'],'actual':item['key'],'predicted':pred,
            'answer':answer,'raw_mu':mu.tolist(), 'input_tokens':len(ids)}


def audit_native(mso,items):
    audit=[]
    for item in items[:2]:
        for typ in ('choice','noul'):
            custom=decision(mso,item,'full',typ)
            official=mso.ask(item['q']['images'][0],native_question(item,typ))
            fields=('choice','probabilities','abstain','valid') if typ=='choice' else ('noul',)
            if any(custom['answer'][k]!=official[k] for k in fields):
                raise ValueError('Custom full-image path differs from official decision API')
            if typ=='choice':
                swapped=decision(mso,item,'full',typ,reverse=True)
                if swapped['predicted']!=custom['predicted'] or swapped['answer']['probabilities']!=custom['answer']['probabilities']:
                    raise ValueError('Choice order audit failed')
            audit.append({'id':item['id'],'type':typ,'official_equal':True,'swapped_equal':typ=='choice'})
    return audit


def embedding_predictions(items, queries, targets):
    import torch
    bank=torch.stack([targets[r['d']['text']] for r in items])
    q=torch.stack([queries[r['id']] for r in items])
    ranks=(q@bank.T).argmax(-1).tolist()
    return [{'id':r['id'],'actual':r['key'],'predicted':items[j]['key'],
             'candidate_id':items[j]['id']} for r,j in zip(items,ranks)]


def evaluate(arm, limit):
    import torch
    from filelock import FileLock
    from superembedding.checkpoint import atomic_save
    from superembedding.general import GeneralOmniEncoder, GENERAL_BASIS
    from superembedding.omnijev import load_omnijev
    from superembedding.general_inference import load_general_state
    items,protocol=prepare(); sig=signature(protocol)
    arm_dir=OUT/arm; arm_dir.mkdir(exist_ok=True)
    with FileLock(str(arm_dir/'evaluation.lock'),timeout=0):
        if (arm_dir/'result.json').exists():
            if read(arm_dir/'result.json')['protocol_hash']!=sig: raise ValueError('Result protocol mismatch')
            return
        write({'state':'loading','arm':arm,'pid':os.getpid()},OUT/'worker_status.json')
        mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
        model=GeneralOmniEncoder(mso,GENERAL_BASIS,embed_dim=256,mode='raw',tune_lora=False).eval()
        model.requires_grad_(False); mso.model.eval(); mso.model.requires_grad_(False)
        if next(model.parameters()).device.type!='cpu': raise ValueError('CPU isolation failed')
        head_before=tensor_sha(mso.model.head.state_dict())
        # Check the official loader did not silently omit original decision-head tensors.
        official_head=torch.load(ROOT/'models/OmniJev-0.8B-v1.1/head.pt',map_location='cpu',weights_only=True)
        if head_before!=tensor_sha(official_head): raise ValueError('Original decision head load mismatch')
        lora_before=tensor_sha({n:p for n,p in model.backbone.named_parameters() if 'lora_' in n})
        saved=torch.load(STEP500 if arm=='step500' else INITIAL,map_location='cpu',weights_only=True)
        if saved['step']!=(500 if arm=='step500' else 0): raise ValueError('Wrong checkpoint step')
        load_general_state(model,saved)
        lora_after=tensor_sha({n:p for n,p in model.backbone.named_parameters() if 'lora_' in n})
        if (lora_before==lora_after)!=(arm=='original'): raise ValueError('LoRA treatment did not match arm')
        if tensor_sha(mso.model.head.state_dict())!=head_before: raise ValueError('Decision head was modified')
        write({'head_original_exact':True,'head_sha256':head_before,'lora_before':lora_before,
               'lora_after':lora_after,'lm_feats':mso.lm_feats,'temperatures':mso.temps,
               'native_api':audit_native(mso,items)},arm_dir/'audit.json')
        del saved,official_head
        cache=arm_dir/'cache';cache.mkdir(exist_ok=True)

        def feature(obj):
            key=signature({'object':obj,'protocol':sig,'arm':arm})
            path=cache/(key+'.pt')
            if path.exists():
                data=torch.load(path,map_location='cpu',weights_only=True)
                if data['key']!=key: raise ValueError('Feature identity mismatch')
                u=data['u']
            else:
                with torch.no_grad(): u,_=model.extract_features(obj)
                u=u.detach().cpu()
                atomic_save({'key':key,'u':u},path)
            if u.shape!=(4,1024) or not torch.isfinite(u).all(): raise ValueError('Invalid cached hidden')
            return u

        targets={s:feature({'text':s}) for s in sorted({r['d']['text'] for r in items})}
        started=time.monotonic()
        for i,item in enumerate(items[:limit or None],1):
            for mode in ('full','text_only'):
                obj=item['q'] if mode=='full' else {'text':item['q']['text']}
                feature(obj)
                for typ in ('choice','noul'):
                    key=signature({'id':item['id'],'mode':mode,'type':typ,'protocol':sig,'arm':arm})
                    p=cache/(key+'.json')
                    if p.exists():
                        if read(p)['key']!=key: raise ValueError('Decision cache identity mismatch')
                    else:
                        write({'key':key,'prediction':decision(mso,item,mode,typ)},p)
            status={'state':'evaluating','arm':arm,'completed_rows':i,'total':len(items),
                    'seconds':time.monotonic()-started,'pid':os.getpid(),'updated_unix':time.time()}
            write(status,OUT/'worker_status.json')
            if i==1 or i%10==0: print(json.dumps(status),flush=True)
            if (OUT/'STOP').exists(): return
        if limit and limit<len(items):
            write({'state':'smoke_complete','rows':limit,'arm':arm,'not_a_full_metric':True},OUT/'worker_status.json')
            return
        results={}
        for typ in ('choice','noul'):
            for mode in ('full','text_only'):
                preds=[]
                for item in items:
                    key=signature({'id':item['id'],'mode':mode,'type':typ,'protocol':sig,'arm':arm})
                    preds.append(read(cache/(key+'.json'))['prediction'])
                results[typ+'_'+mode]={'metrics':metrics(preds),'predictions':preds}
        with torch.no_grad():
            target_vectors={s:model.embedding_head(u).embedding for s,u in targets.items()}
            for mode in ('full','text_only'):
                queries={r['id']:model.embedding_head(feature(r['q'] if mode=='full' else {'text':r['q']['text']})).embedding for r in items}
                preds=embedding_predictions(items,queries,target_vectors)
                results['embedding_'+mode]={'metrics':metrics(preds),'predictions':preds}
        # Original embedding is the reconstructed random-head baseline, not trained frozen projection.
        write({'protocol_hash':sig,'arm':arm,'results':results,
               'embedding_status':'trained at step500' if arm=='step500' else 'reconstructed random projection; not the frozen-trained arm'},arm_dir/'result.json')
        if tensor_sha(mso.model.head.state_dict())!=head_before: raise ValueError('Decision head changed during evaluation')
        write({'state':'completed','arm':arm,'pid':os.getpid()},OUT/'worker_status.json')


def frozen_ready():
    return all((FROZEN/f'step_000500_{lr}_head.pt').exists() for lr in LRS)


def evaluate_frozen():
    import torch
    from superembedding.heads import OptionEmbeddingHead
    items,protocol=prepare(); sig=signature(protocol)
    if not frozen_ready(): raise ValueError('Matched frozen step500 heads not ready')
    expected=signature(read(FROZEN/'protocol.json'))
    cache=OUT/'original/cache'
    def feature(obj):
        key=signature({'object':obj,'protocol':sig,'arm':'original'})
        saved=torch.load(cache/(key+'.pt'),map_location='cpu',weights_only=True)
        if saved['key']!=key or saved['u'].shape!=(4,1024) or not torch.isfinite(saved['u']).all():
            raise ValueError('Original frozen feature cache invalid')
        return saved['u']
    for lr in LRS:
        path=FROZEN/f'step_000500_{lr}_head.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['step']!=500 or saved['protocol_hash']!=expected: raise ValueError('Frozen snapshot mismatch')
        head=OptionEmbeddingHead(1024,embed_dim=256,mode='raw').eval()
        head.load_state_dict(saved['head'],strict=True)
        result={}
        with torch.no_grad():
            targets={s:head(feature({'text':s})).embedding for s in {r['d']['text'] for r in items}}
            for mode in ('full','text_only'):
                queries={r['id']:head(feature(r['q'] if mode=='full' else {'text':r['q']['text']})).embedding for r in items}
                preds=embedding_predictions(items,queries,targets)
                result['embedding_'+mode]={'metrics':metrics(preds),'predictions':preds}
        write({'protocol_hash':sig,'head_sha256':sha(path),'lr':lr,'step':500,'results':result},OUT/f'frozen_{lr}/result.json')


def report():
    items,protocol=prepare(); sig=signature(protocol)
    arms={}
    for name in ['original','step500']+['frozen_'+lr for lr in LRS]:
        p=OUT/name/'result.json'
        if p.exists():
            data=read(p)
            if data['protocol_hash']!=sig: raise ValueError('Report protocol mismatch')
            for method,result in data['results'].items(): arms[name+'/'+method]=result
    qpath=ROOT/'artifacts/results/v5/validation_controls/qwen2b_cpu.json'
    q=read(qpath)
    if q['provenance']['protocol']!=protocol['validation_protocol']: raise ValueError('Qwen protocol differs')
    labels={r['id']:r['key'] for r in items}
    for mode,key in [('full','full'),('text_only','qa_text_only')]:
        preds=[{'id':p['id'],'actual':labels[p['id']],'predicted':labels[p['candidate_id']]}
               for p in q[key]['predictions'] if p['id'] in labels]
        if {p['id'] for p in preds}!=labels.keys(): raise ValueError('Incomplete Qwen predictions')
        arms['qwen2b/'+mode]={'metrics':metrics(preds),'predictions':preds}
    for label in LABELS:
        preds=[{'id':r['id'],'actual':r['key'],'predicted':label} for r in items]
        arms['always_'+label]={'metrics':metrics(preds),'predictions':preds}
    comparisons={}
    pairs=[('step500/choice_full','original/choice_full'),('step500/noul_full','original/noul_full'),
           ('step500/embedding_full','original/choice_full'),('step500/embedding_full','step500/choice_full'),
           ('step500/embedding_full','qwen2b/full')]
    pairs += [('step500/embedding_full','frozen_'+lr+'/embedding_full') for lr in LRS]
    pairs += [(k,k.removesuffix('full')+'text_only') for k in arms if k.endswith('_full')]
    for a,b in pairs:
        if a in arms and b in arms: comparisons[a+' minus '+b]=paired(arms[a]['predictions'],arms[b]['predictions'])
    complete=all((OUT/n/'result.json').exists() for n in ['original','step500']+['frozen_'+lr for lr in LRS])
    write({'state':'completed' if complete else 'partial','protocol_hash':sig,
           'qwen_result_sha256':sha(qpath),'arms':arms,'paired_comparisons':comparisons,
           'limitations':protocol['limitations']},OUT/'summary.json')
    lines=['# VSR 同协议决策与嵌入对照', '',
           'CPU FP32；110 条内部验证（Yes 68 / No 42），不是官方 VSR 测试集。',
           'original/embedding 是重建随机 head，不是冻结后训练的 projection。',
           'step500 为固定检查点；冻结组同为 500 步，三档 LR 全部报告，不用 VSR 选优。', '',
           '| Model / mode | Correct | Accuracy | Balanced accuracy | Yes recall | No recall | Predicted Yes |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in arms.items():
        m=v['metrics']
        lines.append(f"| {k} | {m['correct']}/110 | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['per_class_recall']['yes']:.2%} | {m['per_class_recall']['no']:.2%} | {m['predicted_yes_rate']:.2%} |")
    lines+=['','状态：'+('全部完成。' if complete else '部分完成；缺失组尚未产出完整结果。'),
            '逐题预测、混淆矩阵、模态差值与配对区间见 summary.json。',
            '决策与嵌入使用各自原生 prompt；不能将二者差异全部归因于 head。']
    (OUT/'SUMMARY.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def pipeline():
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(OUT/'pipeline.lock'),timeout=0):
        for arm in ('original','step500'):
            if (OUT/'STOP').exists(): return
            write({'state':'evaluating','arm':arm,'pid':os.getpid()},OUT/'pipeline_status.json')
            with (OUT/f'{arm}.log').open('a',encoding='utf8') as log:
                subprocess.run([sys.executable,'-B','-u',__file__,'--arm',arm],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-B',__file__,'--report'],cwd=ROOT,check=True)
        deadline=time.monotonic()+96*3600
        while not frozen_ready():
            if (OUT/'STOP').exists():
                write({'state':'paused','phase':'waiting_for_frozen_step500'},OUT/'pipeline_status.json');return
            if time.monotonic()>deadline: raise TimeoutError('Frozen step500 wait exceeded 96 hours')
            state=read(FROZEN/'status.json') if (FROZEN/'status.json').exists() else {}
            write({'state':'waiting_for_matched_frozen_step500','frozen_state':state,
                   'pid':os.getpid(),'updated_unix':time.time()},OUT/'pipeline_status.json')
            time.sleep(30)
        with (OUT/'frozen.log').open('a',encoding='utf8') as log:
            subprocess.run([sys.executable,'-B','-u',__file__,'--frozen'],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        write({'state':'completed','summary':str(OUT/'SUMMARY.zh.md')},OUT/'pipeline_status.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--arm',choices=['original','step500'])
    action.add_argument('--frozen',action='store_true')
    action.add_argument('--report',action='store_true')
    action.add_argument('--pipeline',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();os.chdir(ROOT)
    try:
        if args.pipeline: pipeline()
        else:
            cpu_setup()
            if args.prepare: prepare();report();print('Protocol sealed; cached-reference report ready.',flush=True)
            elif args.arm: evaluate(args.arm,args.limit)
            elif args.frozen: evaluate_frozen();report()
            else: report()
    except Exception as exc:
        write({'state':'failed','phase':vars(args),'error':type(exc).__name__,'message':str(exc),
               'resume':'Inspect failure, rerun same command; cached completed rows are retained.'},
              OUT/('pipeline_status.json' if args.pipeline else 'worker_status.json'))
        raise
