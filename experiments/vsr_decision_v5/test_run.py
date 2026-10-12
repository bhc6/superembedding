"""CPU tests for label scoring, paired alignment, and repeated answer candidates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import unittest
import torch
import run


class VSRTests(unittest.TestCase):
    def test_majority_and_confusion(self):
        p=[{'id':str(i),'actual':'yes' if i<68 else 'no','predicted':'yes'} for i in range(110)]
        m=run.metrics(p)
        self.assertEqual(m['correct'],68)
        self.assertEqual(m['balanced_accuracy'],.5)
        self.assertEqual(m['confusion'],{'yes':{'yes':68,'no':0},'no':{'yes':42,'no':0}})
        with self.assertRaises(ValueError): run.metrics(p+p[:1])

    def test_paired_comparison_uses_identity_and_labels(self):
        a=[{'id':str(i),'actual':'yes' if i<3 else 'no','predicted':'yes' if i<3 else 'no'} for i in range(4)]
        b=[{**p,'predicted':'no' if p['actual']=='yes' else 'yes'} for p in reversed(a)]
        delta=run.paired(a,b)
        self.assertEqual(delta['accuracy_delta'],1.)
        self.assertEqual(delta['balanced_accuracy_delta'],1.)
        self.assertEqual(delta['accuracy_paired_bootstrap_95'],[1.,1.])
        with self.assertRaises(ValueError):run.paired(a,b[:-1])
        with self.assertRaises(ValueError):run.paired(a,[{**p,'actual':'yes'} for p in b])

    def test_repeated_target_candidates_not_diagonal_matching(self):
        items=[{'id':str(i),'key':label,'d':{'text':label+'.'}} for i,label in enumerate(['no','yes','yes'])]
        targets={'no.':torch.tensor([0.,1.]),'yes.':torch.tensor([1.,0.])}
        queries={'0':torch.tensor([0.,1.]),'1':torch.tensor([1.,0.]),'2':torch.tensor([1.,0.])}
        p=run.embedding_predictions(items,queries,targets)
        self.assertEqual(run.metrics(p)['accuracy'],1.)
        self.assertEqual(p[2]['candidate_id'],'1')
        # Exact ties follow first record-id candidate, matching existing validation.
        queries['2']=torch.tensor([1.,1.])
        self.assertEqual(run.embedding_predictions(items,queries,targets)[2]['predicted'],'no')

    def test_native_question_removes_only_training_prefix(self):
        item={'q':{'text':'Find the answer to this question: Is the cat on the chair? Answer yes or no.'}}
        q=run.native_question(item,'choice')
        self.assertEqual(q['instructions'],'Is the cat on the chair? Answer yes or no.')
        self.assertEqual([o['key'] for o in q['options']],['yes','no'])
        self.assertNotIn('options',run.native_question(item,'noul'))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
