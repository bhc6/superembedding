"""Train-only question lookup and fixed seen/unseen-question diagnostic subgroups."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_data_leakage import normalize_text
from superembedding.checkpoint import atomic_json
from superembedding.validation_protocol import load_protocol


def answer_key(text):
    return ' '.join(text.split()).casefold().strip(' .!?')


def train_lookup(rows):
    questions = defaultdict(Counter)
    majority = defaultdict(Counter)
    for r in rows:
        if r['split'] != 'train' or r['task'] != 'qa':
            continue
        for t in r['texts']:
            key = answer_key(t['assistant'])
            questions[(r['source'],normalize_text(t['user']))][key] += 1
            majority[r['source']][key] += 1
    return questions, majority


def most_common(counter):
    return min(counter, key=lambda k:(-counter[k],k))


def main():
    protocol,items,rows = load_protocol('D:/DEMO/superembedding-storage/superembedding/v5/data_v51')
    questions,majority = train_lookup(rows)
    lookup = {r['id']:r for r in rows}
    preds=[]
    for item in items:
        if item['task']!='qa':
            continue
        row=lookup[item['id']]
        t=random.Random(row['id']).choice(row['texts'])
        q=normalize_text(t['user'])
        counter=questions.get((row['source'],q))
        pred=most_common(counter or majority[row['source']])
        preds.append({'id':row['id'],'source':row['source'],'seen_question':bool(counter),
                      'seen_question_and_answer':bool(counter and item['key'] in counter),
                      'conflicting_train_answers':bool(counter and len(counter)>1),
                      'question_sha256':hashlib.sha256(q.encode()).hexdigest(),
                      'lookup_correct':pred==item['key']})
    summary={}
    for source in sorted({p['source'] for p in preds}):
        group=[p for p in preds if p['source']==source]
        summary[source]={'queries':len(group),'seen_question':sum(p['seen_question'] for p in group),
                         'seen_question_and_answer':sum(p['seen_question_and_answer'] for p in group),
                         'lookup_R1':sum(p['lookup_correct'] for p in group)/len(group),
                         'models':{}}
    result_root=Path('artifacts/results/v5/validation_controls')
    # Existing results are summarized; no model, prompt or filter is selected by these metrics.
    for path in sorted(result_root.glob('*.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        if result.get('provenance',{}).get('protocol',{}).get('data_sha256')!=protocol['data_sha256']:
            continue
        for mode in ['full','qa_text_only']:
            predictions=result.get(mode,{}).get('predictions',[])
            if not predictions:
                continue
            hits={p['id']:p['correct'] for p in predictions}
            for source in summary:
                values={}
                for seen in [True,False]:
                    subset=[p for p in preds if p['source']==source and p['seen_question']==seen]
                    if any(p['id'] not in hits for p in subset):
                        raise ValueError('Incomplete cached prediction coverage')
                    values['seen' if seen else 'unseen']={'n':len(subset),'R1':sum(hits[p['id']] for p in subset)/len(subset) if subset else None}
                summary[source]['models'][path.stem+':'+mode]=values
    report={'protocol':protocol,'rule':'Normalize full question including instruction; all training QA pairs; train-only counts and lexicographic ties; source-majority fallback. No fitting on val/test.',
            'limitation':'Question overlap is a shortcut diagnostic, not proof of image leakage; unseen exact wording is not unseen semantics/entity. No original text released.',
            'per_source':summary,'predictions':preds,'selection_use':False}
    atomic_json(report,Path('artifacts/results/v5/question_overlap_audit.json'))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='models'} for k,v in summary.items()}))


if __name__ == '__main__':
    main()
