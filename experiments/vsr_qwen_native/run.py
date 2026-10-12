"""Independent, resumable native Qwen3.5-0.8B VSR evaluation. CPU only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STORE=Path('D:/DEMO/superembedding-storage/superembedding/v5')
OUT=STORE/'vsr_qwen35_native_v1'
MODEL=ROOT/'models/Qwen3.5-0.8B'
sys.path.insert(0,str(ROOT))


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sig(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write(v,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(v,f,indent=2,ensure_ascii=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def parse_answer(text):
    match=re.fullmatch(r'\s*(yes|no)[.!]?\s*',text,flags=re.IGNORECASE)
    return match[1].lower() if match else 'invalid'


def metrics(rows,pred_key):
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate predictions')
    counts=Counter((r['actual'],r[pred_key]) for r in rows)
    recall={a:counts[a,a]/sum(r['actual']==a for r in rows) for a in ('yes','no')}
    correct=sum(counts[a,a] for a in ('yes','no'))
    return {'n':len(rows),'correct':correct,'accuracy':correct/len(rows),
            'balanced_accuracy':sum(recall.values())/2,'per_class_recall':recall,
            'invalid':sum(r[pred_key]=='invalid' for r in rows),
            'confusion':{a:{b:counts[a,b] for b in ('yes','no','invalid')} for a in ('yes','no')}}


def setup_cpu():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise OSError('Priority setup failed')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(20261012);torch.use_deterministic_algorithms(True)


def protocol():
    from superembedding.validation_protocol import load_protocol
    manifest,items,_=load_protocol(STORE/'data_v51')
    items=[r for r in items if r['source']=='vsr']
    if len(items)!=110 or Counter(r['key'] for r in items)!={'yes':68,'no':42}:raise ValueError('Subset changed')
    for r in items:
        if sha(r['q']['images'][0])!=r['image_file_sha256']:raise ValueError('Image changed')
    import transformers,torch
    p={'name':'vsr-qwen35-08b-native-v1','model':read(MODEL/'source_revision.json'),
       'model_files':{f.name:sha(f) for f in MODEL.iterdir() if f.is_file()},
       'script_sha256':sha(__file__),'validation_protocol':manifest,'items_sha256':sig(items),
       'torch':str(torch.__version__),'transformers':transformers.__version__,
       'device':'cpu','dtype':'float32','threads':1,'max_pixels':65536,
       'prompt':'Original sampled question, without embedding retrieval prefix; native chat template, enable_thinking=False.',
       'decoding':{'do_sample':False,'num_beams':1,'max_new_tokens':16,'repetition_penalty':1.0},
       'primary':'Unconstrained greedy generation; exact Yes/No with optional terminal period/exclamation, case-insensitive. Invalid or unfinished responses count wrong.',
       'secondary':'First-token raw LM logit(Yes)-logit(No), canonical single-token labels; zero threshold, ties Yes. Not generated accuracy or calibrated probability.',
       'modalities':['full','text_only'],'training':'None. No OmniJev adapter, special option tokens, or decision head.',
       'limits':'Retrospective internal VSR110, not official test. No prompt/threshold tuning. Upstream contamination unknown.'}
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=p:raise ValueError('Sealed protocol changed')
    if not path.exists():write(p,path)
    return items,p


def summarize(items,p):
    import numpy as np
    result={};allrows={};boots={};y=np.array([r['key']=='yes' for r in items])
    rng=np.random.default_rng(20261012);ip=rng.integers(0,68,(2000,68));inn=rng.integers(0,42,(2000,42))
    for mode in p['modalities']:
        rows=[]
        for item in items:
            path=OUT/'predictions'/(sig({'protocol':sig(p),'id':item['id'],'mode':mode})+'.json')
            if not path.exists():return False
            r=read(path)
            if r['protocol_hash']!=sig(p) or r['id']!=item['id']:raise ValueError('Prediction identity differs')
            rows.append(r)
        allrows[mode]=rows
        scores=np.array([r['yes_minus_no_logit'] for r in rows]);a=scores[y];b=scores[~y]
        matrix=(a[:,None]>b)+.5*(a[:,None]==b)
        boot=matrix[ip[:,:,None],inn[:,None,:]].mean((1,2));boots[mode]=boot
        result[mode]={'generation':metrics(rows,'generated_label'),'candidate_score':metrics(rows,'score_label'),
                      'auc':float(matrix.mean()),'auc_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
                      'yes_score_quartiles':np.quantile(a,[.25,.5,.75]).tolist(),
                      'no_score_quartiles':np.quantile(b,[.25,.5,.75]).tolist()}
    result['image_gain']={'generation_accuracy':result['full']['generation']['accuracy']-result['text_only']['generation']['accuracy'],
                          'auc':result['full']['auc']-result['text_only']['auc'],
                          'auc_paired_bootstrap_95':np.quantile(boots['full']-boots['text_only'],[.025,.975]).tolist(),
                          'resampling':'2000 paired label-stratified query resamples; exploratory, no seed/selection adjustment'}
    write({'protocol':p,'metrics':result,'predictions':allrows},OUT/'result.json')
    lines=['# Qwen3.5-0.8B 原生 VSR110 对照','',
           'CPU FP32，原模型无 OmniJev LoRA，关闭 thinking；110 条内部验证，非官方测试。','',
           '| 输入 | 生成答对 | 生成 Accuracy | 生成 BA | 无效输出 | Yes/No 首 token AUC |',
           '|---|---:|---:|---:|---:|---:|']
    for mode in p['modalities']:
        m=result[mode];g=m['generation']
        lines.append(f"| {mode} | {g['correct']}/110 | {g['accuracy']:.2%} | {g['balanced_accuracy']:.2%} | {g['invalid']} | {m['auc']:.4f} |")
    lines+=['','生成成绩是主指标；首 token Yes/No 分数是独立补充分析，不替代生成答案。',
            '逐题原始文本、token、候选分数、混淆矩阵与区间见 result.json。']
    (OUT/'SUMMARY.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return True


def main(args):
    setup_cpu()
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(OUT/'evaluation.lock'),timeout=0):
        items,p=protocol();ph=sig(p)
        if args.prepare:print('Protocol sealed.',flush=True);return
        if summarize(items,p):write({'state':'completed','result':str(OUT/'result.json')},OUT/'status.json');return
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText,GenerationConfig
        from PIL import Image
        write({'state':'loading','pid':os.getpid()},OUT/'status.json')
        processor=AutoProcessor.from_pretrained(MODEL,max_pixels=65536,local_files_only=True)
        model,info=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        bad={k:info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'] if info.get(k)}
        if bad:raise ValueError('Incomplete native weight load: '+str(bad))
        model.eval().requires_grad_(False)
        if next(model.parameters()).device.type!='cpu' or any('lora_' in n for n,_ in model.named_parameters()):raise ValueError('Wrong treatment')
        label_ids={label:processor.tokenizer.encode(label,add_special_tokens=False) for label in ['Yes','No']}
        if any(len(v)!=1 for v in label_ids.values()):raise ValueError('Canonical labels are not single tokens')
        eos=processor.tokenizer.eos_token_id
        if eos is None:raise ValueError('Missing tokenizer EOS')
        generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=16,
            repetition_penalty=1.0,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id or eos,
            use_cache=True,return_dict_in_generate=True,output_logits=True)
        write({'passed':True,'no_lora':True,'device':'cpu','dtype':'float32','label_ids':label_ids,
               'eos_token_id':eos,'parameters':sum(t.numel() for t in model.parameters()),
               'generation':generation.to_dict()},OUT/'load_audit.json')
        new=0;start=time.monotonic()
        for i,item in enumerate(items,1):
            prefix='Find the answer to this question: '
            if not item['q']['text'].startswith(prefix):raise ValueError('Unexpected question')
            question=item['q']['text'][len(prefix):]
            for mode in p['modalities']:
                key=sig({'protocol':ph,'id':item['id'],'mode':mode});dest=OUT/'predictions'/(key+'.json')
                if dest.exists():
                    if read(dest)['protocol_hash']!=ph:raise ValueError('Cached prediction mismatch')
                    continue
                if (OUT/'STOP').exists():write({'state':'paused','row':i},OUT/'status.json');return
                frames=[]
                if mode=='full':
                    with Image.open(item['q']['images'][0]) as im:frames.append(im.convert('RGB'))
                content=[{'type':'image'} for _ in frames]+[{'type':'text','text':question}]
                prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                inputs=processor(text=[prompt],images=frames or None,return_tensors='pt')
                if ('pixel_values' in inputs)!=(mode=='full'):raise ValueError('Modality input mismatch')
                model.model.rope_deltas=None
                tick=time.monotonic()
                with torch.inference_mode():generated=model.generate(**inputs,generation_config=generation)
                tokens=generated.sequences[0,inputs['input_ids'].shape[1]:].tolist()
                text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
                terminated=bool(tokens and tokens[-1]==eos)
                logits=generated.logits[0][0].float()
                ly=float(logits[label_ids['Yes'][0]]);ln=float(logits[label_ids['No'][0]])
                if not __import__('math').isfinite(ly-ln):raise ValueError('Nonfinite scores')
                row={'protocol_hash':ph,'id':item['id'],'actual':item['key'],'mode':mode,
                     'generated_text':text,'generated_token_ids':tokens,'terminated':terminated,
                     'generated_label':parse_answer(text) if terminated else 'invalid',
                     'yes_logit':ly,'no_logit':ln,'yes_minus_no_logit':ly-ln,
                     'score_label':'yes' if ly>=ln else 'no','input_tokens':inputs['input_ids'].shape[1],
                     'seconds':time.monotonic()-tick}
                write(row,dest);new+=1
                status={'state':'running','row':i,'total_rows':110,'mode':mode,'new_predictions':new,
                        'pid':os.getpid(),'elapsed_seconds':time.monotonic()-start,'updated_unix':time.time()}
                write(status,OUT/'status.json')
                if new==1 or new%10==0:print(json.dumps(status),flush=True)
                if args.max_new and new>=args.max_new:
                    write({'state':'smoke_complete','new_predictions':new,'not_full_result':True},OUT/'status.json');return
        if not summarize(items,p):raise ValueError('Incomplete final coverage')
        write({'state':'completed','result':str(OUT/'result.json'),'pid':os.getpid()},OUT/'status.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--max-new',type=int,default=0)
    args=parser.parse_args();os.chdir(ROOT)
    try:main(args)
    except Exception as exc:
        write({'state':'failed','error':type(exc).__name__,'message':str(exc),'resume':'Same command after inspecting failure; keep prediction cache.'},OUT/'status.json')
        raise
