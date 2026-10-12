"""Pin and inspect official submissions; never label these locally reproduced."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import httpx

def main():
    root=Path('runs/v5/leaderboard');root.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=90,follow_redirects=True) as c:
        revision=c.get('https://huggingface.co/api/spaces/TIGER-Lab/MMEB-Leaderboard').json()['sha']
        base=f'https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard/resolve/{revision}/'
        tree=c.get(f'https://huggingface.co/api/spaces/TIGER-Lab/MMEB-Leaderboard/tree/{revision}?recursive=true').json()
        response=c.get(base+'datasets.py');response.raise_for_status()
        source=response.text
        # Extract only the literal dataset dictionary, never execute downloaded code.
        import ast
        parsed=ast.parse(source)
        node=next(n for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATASETS' for t in n.targets))
        datasets=ast.literal_eval(node.value)
        groups={k:[x for rows in v.values() for x in rows] for k,v in datasets.items()}
        def fetch(entry):
            path=entry['path'];r=c.get(base+path);r.raise_for_status()
            (root/Path(path).name).write_bytes(r.content)
            obj=r.json(); metrics=obj.get('metrics',{}); metrics=dict(metrics)
            if any(k in metrics for k in ['tool','gui','memory']):
                metrics['agent']={**metrics.get('tool',{}),**metrics.get('gui',{}),**metrics.get('memory',{})}
            scores={}; total_sum=0.;total_count=0; coverage=0
            for modality,names in groups.items():
                vals=[]
                for name in names:
                    value=metrics.get(modality,{}).get(name)
                    if isinstance(value,dict):
                        key=('ndcg_linear@5' if modality in {'text','visdoc'} else 'hit@1')
                        if modality=='visdoc' and key not in value:key='ndcg@5'
                        value=value.get(key)
                    if isinstance(value,(float,int)):
                        vals.append(float(value))
                scores[modality]={'coverage':len(vals),'tasks':len(names),'zero_filled_mean':100*sum(vals)/len(names),
                                  'available_mean':100*sum(vals)/len(vals) if vals else None}
                total_sum+=sum(vals);total_count+=len(names);coverage+=len(vals)
            return {'metadata':obj.get('metadata',{}),'file':path,'sha256':hashlib.sha256(r.content).hexdigest(),
                    'coverage':coverage,'task_count':total_count,'overall_zero_filled':100*total_sum/total_count,
                    'scores':scores}
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(fetch,[x for x in tree if x['path'].startswith('scores/') and x['path'].endswith('.json')]))
    rows.sort(key=lambda x:x['overall_zero_filled'],reverse=True)
    out=Path('artifacts/results/v5');out.mkdir(parents=True,exist_ok=True)
    (out/'reported_leaderboard_audit.json').write_text(json.dumps({'revision':revision,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'source':'https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard','status':'upstream submissions, NOT local reproduction',
        'aggregation':'Official definitions; missing tasks count as zero. Rounded only at final mean; official rounds individual tasks first.',
        'models':rows},indent=2),encoding='utf-8')
    for row in rows[:12]:print(row['metadata'].get('model_name'),round(row['overall_zero_filled'],2),row['coverage'],row['task_count'])

if __name__=='__main__':main()
