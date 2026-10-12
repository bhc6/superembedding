"""Cross-split byte/pixel/perceptual/text audit of every local experiment family."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from superembedding.checkpoint import atomic_json,fingerprint

STORE=Path('D:/DEMO/superembedding-storage/superembedding/v5')
OUT=STORE/'leakage_audit';PUBLIC=Path('artifacts/results/v5')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def normalize_text(text):
    return ' '.join(re.sub(r'[^\w\s]',' ',unicodedata.normalize('NFKC',text).casefold()).split())


def informative(text):
    return len(text)>=40 and len(text.split())>=8


def distinct_split_pairs(rows,groups):
    """Every partition boundary, including val/test and source-heldout/test."""
    pairs=set()
    for ids in groups.values():
        for k,i in enumerate(ids):
            for j in ids[k+1:]:
                if rows[i]['split']!=rows[j]['split']:pairs.add(tuple(sorted((i,j))))
    return pairs


def collect():
    families={};manifests={}
    for name in ['fashion600','fashion2400','fashion_expanded','cub_attributes','dtd']:
        root=Path('data')/name;path=root/'records.jsonl'
        raw=path.read_bytes();rows=[json.loads(s) for s in raw.decode('utf-8').splitlines()]
        manifests[name]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
        families[name]=[{**r,'image':str((root/r['image']).resolve()),'source':name} for r in rows]
    path=STORE/'data_v51/records.jsonl';raw=path.read_bytes()
    rows=[json.loads(s) for s in raw.decode('utf-8').splitlines()]
    manifests['v5']={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    import pyarrow.parquet as pq
    science=STORE/'mmeb_scienceqa'
    for i,r in enumerate(pq.read_table(science/'test.parquet').to_pylist()):
        rows.append({'id':f'scienceqa_official:{i}','source':'scienceqa_official','split':'external_scienceqa',
                     'image':str(science/r['qry_img_path']),'task':'qa',
                     'texts':[{'user':r['qry_text'].replace('<|image_1|>',''),'assistant':r['tgt_text'][0]}]})
    urban=STORE/'urban1k'
    for i,r in enumerate(json.loads((urban/'annotations/annotations.json').read_text(encoding='utf-8'))['annotations']):
        rows.append({'id':f'urban1k_official:{i}','source':'urban1k_official','split':'external_urban1k',
                     'image':str(urban/'images'/r['image']),'task':'caption','texts':[{'user':'','assistant':r['caption']}]})
    evidence=Path('data/evidence_diagnostic_v1')
    for rel in sorted({r[k] for r in [json.loads(s) for s in (evidence/'records.jsonl').read_text(encoding='utf-8').splitlines()]
                       for k in ['anchor','relevant_change','irrelevant_change']}):
        rows.append({'id':'evidence:'+rel,'source':'evidence_diagnostic','split':'external_synthetic',
                     'image':str((evidence/rel).resolve()),'texts':[]})
    families['v5']=rows
    return families,manifests


def features(path):
    sha=digest(path);cache=OUT/'features'/f'{sha}.json'
    if cache.exists():return json.loads(cache.read_text(encoding='utf-8'))
    with Image.open(path) as im:
        im=im.convert('RGB');rgb=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
        basis=np.cos(np.pi*(np.arange(32)+.5)*np.arange(8)[:,None]/32)
        low=(basis@gray@basis.T).flatten()[1:];bits=low>np.median(low)
        ph=sum(int(b)<<i for i,b in enumerate(bits))
        im.thumbnail((640,640),Image.Resampling.LANCZOS)
        small=np.asarray(im.convert('L').resize((9,8)))
        dh=int(''.join('1' if v else '0' for v in (small[:,1:]>small[:,:-1]).flatten()),2)
    value={'file_sha256':sha,'rgb_sha256':rgb,'phash':ph,'dhash':dh}
    # Different paths can contain identical bytes; return first and serialize in
    # the parent thread to avoid two writers racing on the same .tmp file.
    return value


def audit_family(name,rows,feat):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate record id in '+name)
    exact_groups=defaultdict(list);original_groups=defaultdict(list);integrity=[]
    for i,r in enumerate(rows):
        f=feat[r['image']];exact_groups[f['rgb_sha256']].append(i)
        if r.get('original_rgb_sha256'):original_groups[r['original_rgb_sha256']].append(i)
        expected=r.get('image_file_sha256')
        if expected and expected!=f['file_sha256']:integrity.append(r['id'])
        if name!='v5' and r.get('image_sha256')!=f['rgb_sha256']:integrity.append(r['id'])
    exact=distinct_split_pairs(rows,exact_groups)|distinct_split_pairs(rows,original_groups)
    def identity(i,j):
        return {'a':rows[i]['id'],'a_split':rows[i]['split'],'a_source':rows[i]['source'],
                'b':rows[j]['id'],'b_split':rows[j]['split'],'b_source':rows[j]['source']}
    near=[];splits=sorted({r['split'] for r in rows})
    ph=np.array([feat[r['image']]['phash'] for r in rows],dtype=np.uint64)
    dh=np.array([feat[r['image']]['dhash'] for r in rows],dtype=np.uint64)
    for si,sa in enumerate(splits):
        ai=[i for i,r in enumerate(rows) if r['split']==sa]
        for sb in splits[si+1:]:
            bj=np.array([i for i,r in enumerate(rows) if r['split']==sb],dtype=np.int64)
            for i in ai:
                pd=np.bitwise_count(ph[bj]^ph[i]);dd=np.bitwise_count(dh[bj]^dh[i])
                for offset in np.flatnonzero((pd<=6)&(dd<=6)):
                    j=int(bj[offset])
                    if tuple(sorted((i,j))) in exact:continue
                    with Image.open(rows[i]['image']) as a,Image.open(rows[j]['image']) as b:
                        aa=np.asarray(a.convert('RGB').resize((32,32)),dtype=np.float32)
                        bb=np.asarray(b.convert('RGB').resize((32,32)),dtype=np.float32)
                    mae=float(np.abs(aa-bb).mean()/255)
                    near.append({**identity(i,j),'phash_distance':int(pd[offset]),'dhash_distance':int(dd[offset]),
                                 'rgb32_mae':mae,'interpretation':'candidate, not confirmed duplicate'})
    text_groups=defaultdict(set)
    for i,r in enumerate(rows):
        for t in r.get('texts',[]):
            answer=normalize_text(t['assistant']);question=normalize_text(t['user'])
            # Shared yes/no/numeric targets and question templates are not treated
            # as copied examples. Longer caption and complete QA matches flagged.
            if r.get('task')=='caption' and informative(answer):key='caption:'+answer
            elif r.get('task')=='qa' and informative(question):key='qa:'+question+'\n'+answer
            else:continue
            text_groups[hashlib.sha256(key.encode()).hexdigest()].add(i)
    text=[]
    for h,group in text_groups.items():
        group=sorted(group)
        if len({rows[i]['split'] for i in group})>1:
            text.append({'text_sha256':h,'members':[{'id':rows[i]['id'],'split':rows[i]['split']} for i in group],
                         'interpretation':'Shared full QA/caption needs review; repeated generic QA can be legitimate across distinct images'})
    components=defaultdict(set)
    for r in rows:
        if 'duplicate_component' in r:components[r['duplicate_component']].add(r['split'])
    return {'records':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
            'image_integrity_failures':integrity,'exact_cross_split_pairs':[identity(i,j) for i,j in sorted(exact)],
            'perceptual_candidates':sorted(near,key=lambda x:x['rgb32_mae']),
            'long_text_cross_split_groups':text,'duplicate_components_spanning_splits':[k for k,v in components.items() if len(v)>1],
            'limits':'Full-image hashes do not exhaust crops, alternate views, same product/document/scene or paraphrases; pretraining data and provider training are not observable.'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    families,manifests=collect();paths=sorted({r['image'] for rows in families.values() for r in rows})
    atomic_json({'state':'extracting','images':len(paths)},OUT/'status.json')
    feat={}
    with ThreadPoolExecutor(4) as pool:
        for i,(path,value) in enumerate(zip(paths,pool.map(features,paths)),1):
            feat[path]=value;cache=OUT/'features'/f"{value['file_sha256']}.json"
            if not cache.exists():atomic_json(value,cache)
            if i%1000==0:
                atomic_json({'state':'extracting','completed':i,'images':len(paths)},OUT/'status.json')
                print(json.dumps({'images':i,'total':len(paths)}),flush=True)
    results={}
    # Audit the active training family first. Never mutate an active manifest.
    for name in ['v5','fashion600','fashion2400','fashion_expanded','cub_attributes','dtd']:
        atomic_json({'state':'screening','family':name},OUT/'status.json')
        results[name]=audit_family(name,families[name],feat)
        atomic_json(results[name],OUT/f'{name}.json')
        r=results[name]
        print(json.dumps({'family':name,'exact_pairs':len(r['exact_cross_split_pairs']),
                          'near_candidates':len(r['perceptual_candidates']),'text_groups':len(r['long_text_cross_split_groups'])}),flush=True)
    # Previous pilot exposure is excluded from the v3/v4 validation/test boundary.
    old={feat[r['image']]['rgb_sha256'] for name in ['fashion600','fashion2400'] for r in families[name]}
    pilot_overlap=[r['id'] for r in families['fashion_expanded'] if r['split']!='train' and feat[r['image']]['rgb_sha256'] in old]
    provenance={'utc':datetime.now(timezone.utc).isoformat(),'manifests':manifests,'script_sha256':digest(__file__),
                'method':'Decoded full RGB and recorded original RGB exact matching; fresh JPEG DCT63 pHash<=6 AND dHash64<=6; all split boundaries; normalized long caption or full QA equality',
                'prediction_or_metric_used_for_filtering':False}
    report={'provenance':provenance,'families':results,'pilot_rgb_in_expanded_held':pilot_overlap,
            'pretraining_contamination':'Unknown for inherited checkpoints and proprietary API; cannot certify absence.',
            'all_experiments_leak_free':False,
            'selection_policy':'All seen benchmark results are exploratory for subsequent design changes. Validation can select checkpoints; test cannot select prompts, architecture, LR, data or stopping step.',
            'coverage':'V1/V2 pilot datasets; shared V3/V4 datasets; V5 train/val/ood and all external image sets. CPU head grid and queued frozen-LoRA share audited V5 manifest.'}
    atomic_json(report,PUBLIC/'data_leakage_audit.json')
    lines=['# 全实验数据泄露审计','',provenance['utc'],'',
           '本文件仅为自动筛查。已确认的同图/同实体问题与当前阻断状态见 [人工式视觉复核与处置](DATA_LEAKAGE_REVIEW.zh.md)；候选计数不能代表已排除泄露。','',
           '| 实验数据 | 图像数 | 跨划分精确重复对 | 近重复候选对 | 长文本重复组 |','|---|---:|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['records']} | {len(r['exact_cross_split_pairs'])} | {len(r['perceptual_candidates'])} | {len(r['long_text_cross_split_groups'])} |")
    lines += ['', '近重复候选不是确认泄露；图表模板、纹理和合成图像可能产生误报。长文本相同也可能是不同图像的通用问答，不能等同于答案泄露。必须逐项复核，不按模型对错选择删除样本。','',
              '所有现有公开测试已被观察，后续针对弱项调整的实验只能算探索性。Choice 只使用验证集，仍不是未见任务证据；新的确认性结论需未接触且按对象/文档/场景分组隔离的数据。','',
              '原始 backbone/OmniJev/FLAIR/Qwen 等权重及 API 的预训练污染无法由本地审计排除，不可宣称零污染。','',
              '完整候选清单用 ID/哈希记录，保留在 JSON；不含原始问答文本。原 manifests、checkpoint 和历史结果不覆盖。','']
    Path('reports/DATA_LEAKAGE_V5.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json({'state':'screening_completed_review_required','report':'artifacts/results/v5/data_leakage_audit.json',
                 'counts':{k:{field:len(v[field]) for field in ['exact_cross_split_pairs','perceptual_candidates','long_text_cross_split_groups']} for k,v in results.items()}},OUT/'status.json')


if __name__=='__main__':main()
