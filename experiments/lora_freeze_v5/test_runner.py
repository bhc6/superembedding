"""CPU-only checks of matched sampling, cached gradients, and interruption replay."""
# All tensors in these tests are explicitly constructed on CPU. Do not change
# CUDA visibility during pytest collection: other modules already queried CUDA,
# and changing it here can prevent their checkpoint helpers saving GPU RNG.
import io
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('freeze_runner', Path(__file__).with_name('runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.modules['runner'] = runner
pipeline_spec = importlib.util.spec_from_file_location('freeze_pipeline', Path(__file__).with_name('pipeline.py'))
pipeline = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(pipeline)
from scripts.train_general import pair
from superembedding.general import cached_backward, positive_mask
from superembedding.heads import OptionEmbeddingHead


def assert_tree_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a,b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a: assert_tree_equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): assert_tree_equal(test,x,y)
    else:
        test.assertEqual(a,b)


class MatchedTrainingTests(unittest.TestCase):
    def test_paired_comparison_identity_macro_weighting_and_intervals(self):
        a = [{'id': str(i), 'correct': i == 0} for i in range(4)]
        b = [{'id': str(i), 'correct': False} for i in reversed(range(4))]
        result = pipeline.paired_summary(a, b, 'id', {'small':['0'], 'large':['1','2','3']})
        self.assertEqual(result['lora_minus_frozen'], .5)
        self.assertEqual(result['paired_bootstrap_95'], [.5, .5])
        with self.assertRaises(ValueError):
            pipeline.paired_summary(a, b[:-1], 'id')
        with self.assertRaises(ValueError):
            pipeline.paired_summary(a, b, 'id', {'missing':['0']})

    def test_schedule_matches_original_sampling_including_curriculum(self):
        rows=[]
        for s in ['a','b']:
            for i in range(20):
                rows.append({'id':s+str(i),'source':s,'split':'train','image':s+str(i)+'.jpg',
                             'order':i if i<12 else 256+i,'task':'qa',
                             'texts':[{'user':'question','assistant':str(j)} for j in range(i%4+1)]})
        config={'seed':17,'steps':510,'batch_size':4}
        rng=random.Random(17);queue=[]
        for item,state,remaining in runner.schedule(rows,config):
            if not queue:queue=['a','b'];rng.shuffle(queue)
            source=queue.pop()
            pool=[r for r in rows if r['source']==source and (item['step']>500 or r['order']<256)]
            for _ in range(32):
                batch=rng.sample(pool,4)
                pairs=[pair(r,rng) for r in batch]
                if not positive_mask([p[2] for p in pairs],'cpu').all():break
            self.assertEqual(source,item['source'])
            self.assertEqual([r['id'] for r in batch],[i[0] for i in item['samples']])
            self.assertEqual(rng.getstate(),state)
            self.assertEqual(queue,remaining)
            for r,(_,idx),p in zip(batch,item['samples'],pairs):
                self.assertEqual(p,runner.objects_for(r,r['texts'][idx]))

    def test_feature_cache_matches_frozen_encoder_and_exact_resume(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        backbone=torch.nn.Linear(12,16).requires_grad_(False).eval()
        q=[torch.randn(4,12) for _ in range(4)]
        d=[torch.randn(4,12) for _ in range(4)]
        uq=[backbone(x).detach() for x in q];ud=[backbone(x).detach() for x in d]
        head=OptionEmbeddingHead(16,embed_dim=4,mode='raw')
        cached=deepcopy(head)
        keys=['same','same','b','c']
        cached_backward(lambda x:head(backbone(x)).embedding,q,d,keys)
        cached_backward(lambda x:cached(x).embedding,uq,ud,keys)
        for a,b in zip(head.parameters(),cached.parameters()):
            self.assertTrue(torch.equal(a.grad,b.grad))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        original=deepcopy(cached.state_dict())

        def build():
            h=OptionEmbeddingHead(16,embed_dim=4,mode='raw');h.load_state_dict(original)
            return h,torch.optim.AdamW(h.parameters(),lr=1e-5,weight_decay=.01)

        def updates(h,opt,start,end):
            for step in range(start,end+1):
                opt.zero_grad(set_to_none=True)
                _,error=cached_backward(lambda x:h(x).embedding,uq,ud,keys)
                self.assertEqual(error,0)
                torch.nn.utils.clip_grad_norm_(h.parameters(),1.)
                for group in opt.param_groups:group['lr']=1e-5*runner.lr_scale(step,2000)
                opt.step()

        full,fo=build();updates(full,fo,1,6)
        partial,po=build();updates(partial,po,1,3)
        memory=io.BytesIO();torch.save({'head':partial.state_dict(),'optimizer':po.state_dict()},memory)
        memory.seek(0);saved=torch.load(memory,weights_only=True)
        resumed,ro=build();resumed.load_state_dict(saved['head']);ro.load_state_dict(saved['optimizer'])
        updates(resumed,ro,4,6)
        assert_tree_equal(self,full.state_dict(),resumed.state_dict())
        assert_tree_equal(self,fo.state_dict(),ro.state_dict())


if __name__=='__main__':
    unittest.main()
