"""Independent official-API and backbone spot checks; cached outputs are comparison only."""
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.vsr_decision_v5 import run as ref


def main():
    ref.cpu_setup()
    import torch
    from superembedding.omnijev import load_omnijev
    from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS
    from superembedding.general_inference import load_general_state
    items,protocol=ref.prepare();sig=ref.signature(protocol)
    output=ref.STORE/'vsr_fresh_verification_v1';output.mkdir(exist_ok=True)
    # Fixed positional sample, decided before seeing prediction differences.
    positions=[0,len(items)//2,len(items)-1]
    selected=[items[i] for i in positions]
    plan={'indices':positions,'ids':[r['id'] for r in selected],
          'selection':'First/middle/last in sealed record-ID order; no outcome-based selection',
          'protocol_hash':sig,'script_sha256':ref.sha(__file__),
          'fresh_original':'Official loader and ask API directly, no reconstructed checkpoint',
          'fresh_step500':'Official loader, then all immutable Step500 trained tensors; original decision head unchanged',
          'max_pixels':65536,'device':'CPU FP32 one thread','tolerances':{'embedding_max_abs':1e-6,'rounded_noul':0},
          'api_requeried':False}
    if (output/'plan.json').exists() and ref.read(output/'plan.json')!=plan:raise ValueError('Plan changed')
    ref.write(plan,output/'plan.json')
    rows={r['id']:r for r in map(json.loads,(ref.STORE/'data_v51/records.jsonl').read_text(encoding='utf-8').splitlines())}
    for item in items:
        row=rows[item['id']]
        source=ref.read(ref.STORE/'data/rows'/f"vsr_{row['source_row']}.json")
        assert source['texts']==row['texts'] and source['original_rgb_sha256']==row['original_rgb_sha256']
        text=random.Random(row['id']).choice(source['texts'])
        assert item['key']==' '.join(text['assistant'].split()).casefold().strip(' .!?')
        assert item['q']['text']=='Find the answer to this question: '+text['user']
        assert set(item['q'])=={'text','images'}
    checks=[]
    mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
    mso.model.eval();mso.model.requires_grad_(False)
    original_head=ref.tensor_sha(mso.model.head.state_dict())
    official_head=torch.load('models/OmniJev-0.8B-v1.1/head.pt',map_location='cpu',weights_only=True)
    assert original_head==ref.tensor_sha(official_head)
    assert next(mso.model.parameters()).device.type=='cpu'
    for arm in ['original','step500']:
        cached=ref.read(ref.OUT/arm/'result.json')['results']
        if arm=='step500':
            model=GeneralOmniEncoder(mso,GENERAL_BASIS,embed_dim=256,mode='raw',tune_lora=False).eval()
            saved=torch.load(ref.STEP500,map_location='cpu',weights_only=True)
            assert saved['step']==500
            load_general_state(model,saved)
            assert original_head==ref.tensor_sha(mso.model.head.state_dict())
            del saved
            # All distinct answer forms, encoded fresh; retain original candidate order.
            with torch.no_grad():targets={s:model({'text':s}).embedding for s in sorted({r['d']['text'] for r in items})}
        for item in selected:
            q={'type':'noul','instructions':item['q']['text'].removeprefix('Find the answer to this question: ')}
            answer=mso.ask(item['q']['images'][0],q)
            old=next(p for p in cached['noul_full']['predictions'] if p['id']==item['id'])
            record={'arm':arm,'id':item['id'],'fresh_probability':answer['noul'],'cached_probability':old['answer']['noul'],
                    'official_noul_exact':answer['noul']==old['answer']['noul']}
            if not record['official_noul_exact']:raise ValueError('Fresh official noul differs: '+item['id'])
            if arm=='step500':
                for mode in ['full','text_only']:
                    obj=item['q'] if mode=='full' else {'text':item['q']['text']}
                    with torch.no_grad():fresh=model(obj).embedding
                    key=ref.signature({'object':obj,'protocol':sig,'arm':arm})
                    hidden=torch.load(ref.OUT/arm/'cache'/(key+'.pt'),map_location='cpu',weights_only=True)
                    with torch.no_grad():previous=model.embedding_head(hidden['u']).embedding
                    error=float((fresh-previous).abs().max())
                    bank=torch.stack([targets[r['d']['text']] for r in items])
                    predicted=items[int((fresh@bank.T).argmax())]['key']
                    old=next(p for p in cached['embedding_'+mode]['predictions'] if p['id']==item['id'])
                    assert error<=1e-6 and predicted==old['predicted']
                    record[mode]={'fresh_prediction':predicted,'cached_prediction':old['predicted'],'embedding_max_abs':error}
            checks.append(record)
            ref.write({'state':'running','checks':checks},output/'status.json')
            print(json.dumps(record),flush=True)
    assert original_head==ref.tensor_sha(mso.model.head.state_dict())
    report={'passed':True,'plan':plan,'checks':checks,'source_annotations_checked':110,
            'query_ground_truth_field_injected':False,'unique_images':len({r['image_file_sha256'] for r in items}),
            'limits':'Fresh inference spot-check, not all-110 fresh rerun. Qwen vectors replayed separately; API was not requeried. No claim of leakage-free generalization.'}
    ref.write(report,output/'audit.json')
    ref.write(report,ref.ROOT/'artifacts/results/v5/vsr_fresh_verification.json')
    ref.write({'state':'completed','passed':True,'checks':len(checks)},output/'status.json')


if __name__=='__main__':main()
