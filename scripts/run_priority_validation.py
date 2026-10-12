"""Finite GPU time sharing: exact training pause/resume around priority controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json


def main():
    root=Path(__file__).resolve().parents[1]
    store=Path('D:/DEMO/superembedding-storage/superembedding/v5')
    run=store/'general_stage1_v51';out=store/'priority_validation';out.mkdir(exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/'pipeline.lock'),timeout=0):
        marker=run/'STOP_AFTER_CHECKPOINT'
        if marker.exists():raise RuntimeError('Existing pause is not owned by this pipeline')
        status=json.loads((run/'status.json').read_text(encoding='utf-8'))
        if status['state']!='running':raise RuntimeError('This driver requires active training to pause and resume')
        selected=store/'validation_selected_step250/general_model.pt'
        audit=json.loads(selected.with_suffix('.audit.json').read_text(encoding='utf-8'))
        if not audit['passed'] or audit['step']!=250 or audit['export_sha256']!=hashlib.sha256(selected.read_bytes()).hexdigest():
            raise ValueError('Selected checkpoint export not verified')
        seal={'selected_step':250,'selection':'First full validation checkpoint; interim, not final 2000-step best',
              'checkpoint_sha256':audit['checkpoint_sha256'],'export_sha256':audit['export_sha256'],
              'test_used_for_selection':False,'pid':os.getpid()}
        atomic_json(seal,out/'selection_sealed.json')
        owner=f'priority-validation:{os.getpid()}'
        marker.write_text(owner,encoding='utf-8')
        paused=False;outcome='failed'
        try:
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                status=json.loads((run/'status.json').read_text(encoding='utf-8'))
                supervisor=json.loads((run/'supervisor.json').read_text(encoding='utf-8'))
                if status.get('state')=='paused' and supervisor.get('state')=='paused':
                    paused=True;break
                if status.get('state')=='failed':raise RuntimeError('Training failed while pausing')
                time.sleep(2)
            if not paused:raise RuntimeError('Training did not confirm pause; no second CUDA model launched')
            atomic_json({'state':'paused_for_evaluation','training_step':status['step'],**seal},out/'status.json')
            init=store/'step0_reconstructed_cuda'
            commands=[]
            if not (init/'general_model.pt').exists():
                commands.append(['scripts/reconstruct_general_step0.py','--device','cuda','--output',str(init)])
            commands.extend([
                ['scripts/evaluate_initial_validation.py','--device','cuda','--checkpoint',str(init/'general_model.pt'),
                 '--audit',str(init/'reconstruction.json')],
                ['scripts/evaluate_validation_controls.py','--backend','general','--device','cuda','--checkpoint',str(selected)],
            ])
            for i,command in enumerate(commands):
                atomic_json({'state':'evaluating','stage':i,'command':command,'training_paused_step':status['step'],**seal},out/'status.json')
                with (out/f'stage_{i}.log').open('a',encoding='utf-8') as log:
                    completed=subprocess.run([sys.executable,'-u',*command],cwd=root,stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:raise RuntimeError(f'Priority stage {i} failed; checkpoint preserved and training will resume')
            outcome='completed'
        finally:
            if marker.exists() and marker.read_text(encoding='utf-8')==owner:
                marker.unlink()
            # Only restart once the prior supervisor confirmed a clean pause.
            if paused:
                with (out/'resumed_supervisor_stdout.log').open('a',encoding='utf-8') as stdout, (out/'resumed_supervisor_stderr.log').open('a',encoding='utf-8') as stderr:
                    process=subprocess.Popen([sys.executable,'-u','scripts/supervise_general.py','--data',str(store/'data_v51'),
                        '--output',str(run),'--steps','2000'],cwd=root,stdout=stdout,stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                atomic_json({'state':outcome,'training_resume_supervisor_pid':process.pid,
                             'resumed_from_step':status['step'],**seal},out/'status.json')
            else:
                atomic_json({'state':'failed_before_confirmed_pause','training_status':status,**seal},out/'status.json')


if __name__=='__main__':main()
