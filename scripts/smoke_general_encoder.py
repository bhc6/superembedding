"""Real CUDA forward/backward smoke, including text-only and mixed objects."""
import os
os.environ['MSO_FLA']='0'; os.environ['MSO_BRANCH_CHUNK']='4'
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from superembedding.general import GeneralOmniEncoder,GENERAL_BASIS,cached_backward
from superembedding.omnijev import load_omnijev
torch.set_num_threads(4); torch.manual_seed(17)
mso=load_omnijev('models/OmniJev-0.8B-v1.1','models/Qwen3.5-0.8B',max_pixels=65536)
model=GeneralOmniEncoder(mso,GENERAL_BASIS,mode='raw',tune_lora=True); model.eval()
paths=list(Path('C:/Users/123/.cache/superembedding/v5/data/images').glob('*.jpg'))[:2]
queries=[{'text':'Find a picture of a person.'},{'images':[str(paths[1])],'text':'What is visible in this image?'}]
targets=[{'images':[str(paths[0])]},{'text':'An outdoor scene with objects.'}]
torch.cuda.reset_peak_memory_stats()
loss,error=cached_backward(lambda obj:model(obj).embedding,queries,targets,['a','b'])
lora=[p.grad.float().norm().item() for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None]
print(json.dumps({'loss':loss,'replay_error':error,'nonzero_lora_gradients':sum(x>0 for x in lora),
    'total_lora_gradients':len(lora),'peak_MiB':torch.cuda.max_memory_allocated()/2**20}),flush=True)
