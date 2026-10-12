"""Fixed-direction image/text memory encoding and replayed contrastive gradients."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from PIL import Image
from .omnijev import OmniJevEmbeddingEncoder
from .heads import AttributeBasis

GENERAL_BASIS = AttributeBasis((
    'Entities, objects and semantic meaning',
    'Appearance, attributes and material',
    'Spatial relations, layout and composition',
    'Written text, numbers and quantitative evidence',
), 'Represent the supplied content for retrieval along each fixed evidence direction.')


class GeneralOmniEncoder(OmniJevEmbeddingEncoder):
    """Candidates use the same fixed basis, independently of any future query."""
    def _branch_features(self, state, enable_grad=False):
        from mso import branch as BR
        text = state.get('text', '')
        for marker in ('<|opt|>', '<|/opt|>', '<|im_start|>', '<|im_end|>'):
            text = text.replace(marker, '')
        paths = state.get('images', [])
        if not text.strip() and not paths:
            raise ValueError('An object must contain text or an image')
        if len(paths) > 1:
            raise ValueError('V5 initial protocol supports one image per object')
        mso = self._mso
        mso.dev = next(self.backbone.parameters()).device
        # Bound text before rendering; never truncate vision/option delimiters.
        tokens = mso.proc.tokenizer.encode(text, add_special_tokens=False)[:256]
        text = mso.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        frames = []
        for path in paths:
            with Image.open(path) as im:
                frames.append(im.convert('RGB'))
        content = [{'type': 'image'} for _ in frames]
        content.append({'type': 'text', 'text': self.basis.instruction + '\nContent:\n' + text})
        prompt = mso.proc.apply_chat_template([{'role': 'user', 'content': content}],
                                             tokenize=False, add_generation_prompt=True)
        prompt += ''.join('<|opt|>' + d + '<|/opt|>' for d in self.basis.directions)
        encoded = mso.proc(text=[prompt], images=frames or None, return_tensors='pt')
        encoded = {k: v.to(mso.dev) for k, v in encoded.items()}
        ids = encoded['input_ids'][0].tolist()
        opens, closes = mso._spans_of(ids)
        if len(opens) != len(self.basis.directions) or len(closes) != len(opens):
            raise ValueError('Unexpected option tokenization')
        prefix_length = max(1, opens[0] - 1)
        with nullcontext() if enable_grad else torch.no_grad():
            penc = BR.prefix_inputs(encoded, prefix_length)
            positions = BR.prefix_positions(mso.model, penc)
            rows = BR.rows_from_ids(ids, prefix_length, opens, closes, k=len(opens))
            [(u, zq, _)] = BR.branch_questions(self.backbone, penc, [rows], mso.dev,
                mso.open_ids, mso.close_ids, prefix_pos=positions, pad_id=mso.pad_id)
        return u, zq


def positive_mask(keys, device):
    def group(key):return set(key) if isinstance(key,(tuple,list,set,frozenset)) else {key}
    groups=[group(key) for key in keys]
    return torch.tensor([[bool(a & b) for b in groups] for a in groups], device=device, dtype=torch.bool)


def contrastive_loss(q, d, keys, temperature=0.05):
    """Directional multi-positive InfoNCE; identical targets are never negatives."""
    scores = q.float() @ d.float().T / temperature
    mask = positive_mask(keys, scores.device)
    return (torch.logsumexp(scores, -1) - torch.logsumexp(scores.masked_fill(~mask, -torch.inf), -1)).mean()


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def cached_backward(encode, queries, targets, keys, temperature=0.05):
    """Exact loss-gradient replay, one object's activations at a time.

    Saves per-object torch RNG for LoRA dropout; model and optimizer must remain
    unchanged between cache and replay. No queues or stale embeddings are used.
    """
    objects = list(queries) + list(targets)
    states, vectors = [], []
    with torch.no_grad():
        for obj in objects:
            states.append(rng_state())
            vectors.append(encode(obj).detach())
    after = rng_state()
    cached = torch.stack(vectors).requires_grad_(True)
    n = len(queries)
    loss = contrastive_loss(cached[:n], cached[n:], keys, temperature)
    grads, = torch.autograd.grad(loss, cached)
    max_error = 0.
    try:
        for obj, state, expected, grad in zip(objects, states, vectors, grads):
            set_rng(state)
            actual = encode(obj)
            max_error = max(max_error, (actual.detach() - expected).abs().max().item())
            torch.sum(actual * grad).backward()
    finally:
        set_rng(after)
    if max_error > 5e-4:
        raise RuntimeError(f'Gradient replay mismatch: {max_error}')
    return loss.item(), max_error
