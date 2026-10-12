import copy
import torch
from superembedding.general import cached_backward, contrastive_loss, positive_mask


def test_dropout_replay_matches_full_batch_gradient_and_rng():
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(7,13),torch.nn.Dropout(.3),torch.nn.Linear(13,5))
    other = copy.deepcopy(model)
    items = list(torch.randn(8,7)); keys = ['a','b','b','c']
    initial = torch.get_rng_state()
    full = torch.stack([torch.nn.functional.normalize(model(x),dim=-1) for x in items])
    loss = contrastive_loss(full[:4], full[4:], keys)
    loss.backward(); expected_rng = torch.get_rng_state()
    torch.set_rng_state(initial)
    value, error = cached_backward(lambda x:torch.nn.functional.normalize(other(x),dim=-1),items[:4],items[4:],keys)
    assert abs(value-loss.item()) < 1e-6 and error == 0
    assert torch.equal(expected_rng, torch.get_rng_state())
    for a,b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)


def test_duplicate_targets_are_positive_not_false_negatives():
    q = torch.eye(2); d = torch.eye(2)
    assert contrastive_loss(q,d,['same','same']).item() == 0
    assert contrastive_loss(q,d,['a','b']).item() >= 0


def test_shared_caption_masks_distinct_image_false_negatives():
    mask=positive_mask([('image:1','caption:a'),('image:2','caption:a'),('image:3','caption:b')],'cpu')
    assert mask.tolist()==[[True,True,False],[True,True,False],[False,False,True]]
