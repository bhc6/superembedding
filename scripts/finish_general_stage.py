"""Run frozen final evaluations after training; never tune to these test scores."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];run=Path(args.run)
    from superembedding.leakage_guard import require_promotion_clearance
    require_promotion_clearance()
    status=json.loads((run/'status.json').read_text())
    if status['state']!='completed':raise RuntimeError('Training has not finished')
    checkpoint=run/'best.pt'
    if not checkpoint.exists():raise RuntimeError('No validation-selected best checkpoint')
    import torch
    selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if selected['step']!=status['best_step']:raise ValueError('Best checkpoint/status step mismatch')
    # Old trainer output is a last-step projection without trained LoRA. Preserve it
    # as explicitly non-deployable; leave fingerprinted running training code intact.
    legacy=run/'embedding_head.pt';preserved=run/'last_step_head_only_DO_NOT_DEPLOY.pt'
    if legacy.exists():
        if preserved.exists():raise ValueError('Legacy export quarantine already exists; inspect both')
        legacy.replace(preserved)
    inference=run/'export'/'general_model.pt'
    export_command=['scripts/export_general_model.py','--checkpoint',str(checkpoint),'--output',str(inference),
                    '--device','cuda','--image',str(root/'data/evidence_diagnostic_v1/images/0000_anchor.png')]
    atomic_json({'state':'verifying_export','command':export_command,'test_based_selection':False},run/'final_evaluation.json')
    with (run/'export_verification.log').open('a',encoding='utf-8') as log:
        exported=subprocess.run([sys.executable,'-u',*export_command],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    if exported.returncode:
        atomic_json({'state':'export_failed','exit_code':exported.returncode},run/'final_evaluation.json')
        raise RuntimeError('Complete model export/reload must pass before formal evaluation')
    audit=json.loads(inference.with_suffix('.audit.json').read_text())
    if not audit['passed'] or audit['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError('Export audit does not match selected best checkpoint')
    tag='general_'+hashlib.sha256(inference.read_bytes()).hexdigest()[:12]
    commands=[['scripts/seal_external_inputs.py','--output',str(run/'export'),'--phase','before'],
              ['scripts/evaluate_validation_controls.py','--backend','general','--device','cuda','--checkpoint',str(inference)],
              ['scripts/report_validation_controls.py'],
              ['scripts/evaluate_mmeb_scienceqa.py','--backend','general','--checkpoint',str(inference)],
              ['scripts/evaluate_mmeb_scienceqa.py','--backend','general','--checkpoint',str(inference),'--query-mode','text_only'],
              ['scripts/evaluate_general_transfer.py','--checkpoint',str(inference)],
              ['scripts/evaluate_general_transfer.py'],
              ['scripts/report_evidence_comparison.py','--models','qwen3_vl','clip_b32','siglip2_b16',
               'flair_global','flair_native','omnijev_raw',tag],
              ['scripts/seal_external_inputs.py','--output',str(run/'export'),'--phase','after'],
              ['scripts/report_general_progress.py']]
    for i,command in enumerate(commands):
        atomic_json({'state':'evaluating','task':i,'command':command,'test_based_selection':False},run/'final_evaluation.json')
        with (run/f'final_evaluation_{i}.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',*command],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json({'state':'failed','task':i,'exit_code':result.returncode},run/'final_evaluation.json')
            raise RuntimeError('Final evaluation failed; preserve checkpoint and inspect log')
    atomic_json({'state':'completed','checkpoint_selection':'validation only','tasks':len(commands),
                 'evaluated_export':str(inference),'export_sha256':audit['export_sha256'],
                 'selected_checkpoint_sha256':audit['checkpoint_sha256'],'selected_step':audit['step'],
                 'publication':'not published; broader reproduction and study remain incomplete'},run/'final_evaluation.json')

if __name__=='__main__':main()
