"""Conservative full-manifest development exposure, not sampled-batch estimates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_data_leakage import normalize_text
from superembedding.checkpoint import atomic_json
from superembedding.leakage_guard import digest


def main():
    root=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    rows={}; memberships=defaultdict(list); manifests=[]
    for folder,run in [('data','general_stage1'),('data_v51','general_stage1_v51')]:
        path=root/folder/'records.jsonl'
        config=json.loads((root/run/'config.json').read_text())
        if digest(path)!=config['data_sha256']:raise ValueError('Historical manifest changed')
        manifests.append({'path':str(path),'sha256':digest(path),'run':run})
        for row in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
            # OOD was also observed diagnostically; conservatively reserve it.
            if row['id'] in rows and rows[row['id']]['image_file_sha256']!=row['image_file_sha256']:
                raise ValueError('Conflicting historical record identity')
            rows[row['id']]=row
            memberships[row['id']].append({'version':folder,'split':row['split']})
    def hashed(value):return hashlib.sha256(normalize_text(value).encode()).hexdigest()
    entries=[]
    for key,row in sorted(rows.items()):
        if digest(row['image'])!=row['image_file_sha256']:raise ValueError('Historical image changed')
        entries.append({'id':key,'source':row['source'],'source_row':row['source_row'],
                        'source_revision':row['source_revision'],'original_rgb_sha256':row['original_rgb_sha256'],
                        'image_file_sha256':row['image_file_sha256'],'memberships':memberships[key],
                        'question_sha256':sorted({hashed(t['user']) for t in row['texts']}),
                        'full_qa_sha256':sorted({hashed(t['user']+'\n'+t['assistant']) for t in row['texts']}),
                        'caption_sha256':sorted({hashed(t['assistant']) for t in row['texts']}) if row['task']=='caption' else []})
    report={'protocol':'v5-development-exposure-union-v1','manifests':manifests,'records':len(entries),
            'policy':'Full historical train/val/ood union, not only sampled training examples. Existing external tests additionally remain seen diagnostics.',
            'limits':'Image/text hashes are inventory keys; independent holdouts still require crop/scene/entity review. Not a leakage-clearance certificate.',
            'entries':entries}
    out=Path('artifacts/results/v5/development_exposure_union.json')
    if out.exists() and json.loads(out.read_text())!=report:raise ValueError('Exposure inventory changed')
    atomic_json(report,out)
    print(json.dumps({'records':len(entries),'file':str(out),'sha256':digest(out)}))


if __name__=='__main__':main()
