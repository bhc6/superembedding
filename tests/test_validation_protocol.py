import torch
from superembedding.validation_protocol import majority_controls, question_type, score_vectors


def row(i, key, task='qa'):
    return {'id':str(i),'source':'clevr','task':task,'key':key,'type':'count'}


def training(i,answer,split='train'):
    return {'id':str(i),'source':'clevr','task':'qa','split':split,'image':'unused',
            'texts':[{'user':'How many objects?','assistant':answer}]}


def test_majority_cannot_fit_validation_labels_or_restrict_to_candidate_answers():
    data=[training(1,'7'),training(2,'7'),training(3,'3')]+[training(i,'3','val') for i in range(10,30)]
    items=[row(100,'3')]
    controls=majority_controls(items,data)
    assert controls['train_source_majority']['macro_r1']==0
    assert controls['train_type_majority']['macro_r1']==0


def test_duplicate_answers_and_training_tie_order():
    items=[row('a','yes'),row('b','yes'),row('c','no')]
    q=torch.ones(3,2);d=torch.ones(3,2)
    result=score_vectors(items,q,d)
    assert result['macro_r1']==2/3
    assert all(p['candidate_id']=='a' for p in result['predictions'])
    assert abs(result['per_source']['clevr']['random_candidate_expected_R1']-5/9)<1e-7


def test_caption_equivalence_uses_image_or_text_keys():
    items=[row('a',('image:x','caption:a'),'caption'),row('b',('image:x','caption:b'),'caption')]
    result=score_vectors(items,torch.eye(2),torch.eye(2).flip(0))
    assert result['macro_r1']==1


def test_question_type_does_not_take_answer():
    assert question_type('How many blue cubes?','clevr')=='count'
    assert question_type('Is there a sphere?','clevr')=='yes_no'
    assert question_type('What color is the cube?','clevr')=='color'
    assert question_type('The man is above the dog.','vsr')=='spatial_statement_verification'
