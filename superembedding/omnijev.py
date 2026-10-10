"""Frozen-backbone OmniJev integration with a differentiable embedding head."""

from pathlib import Path
from contextlib import nullcontext
import sys

import torch
from torch import nn

from .heads import AttributeBasis, OptionEmbeddingHead


def load_omnijev(decision_checkpoint, base_model, repo=None, **kwargs):
    """Load the public OmniJev adapter using its local repository."""
    repo = Path(repo or Path(__file__).resolve().parents[1] / "repos" / "OmniJev").resolve()
    if not (repo / "mso" / "infer.py").is_file():
        raise FileNotFoundError(f"OmniJev repository not found: {repo}")
    existing = sys.modules.get("mso")
    if existing is not None and Path(existing.__file__).resolve().parent != repo / "mso":
        raise RuntimeError("a different mso package is already imported")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from mso.infer import MSO1

    mso = MSO1(str(decision_checkpoint), str(base_model), **kwargs)
    mso._superembedding_source = {
        "decision_checkpoint": str(Path(decision_checkpoint).resolve()) if Path(decision_checkpoint).exists() else str(decision_checkpoint),
        "base_model": str(Path(base_model).resolve()) if Path(base_model).exists() else str(base_model),
    }
    return mso


class OmniJevEmbeddingEncoder(nn.Module):
    """Encode one image/video state into a fixed bank of attribute vectors.

    The backbone is frozen by default. tune_lora=True enables only existing
    LoRA parameters, not dense backbone weights. forward() keeps gradients;
    extract_features() and encode() provide no-grad paths.
    """

    def __init__(self, mso, basis: AttributeBasis, embed_dim=256, mode="gated", train_gate=False, tune_lora=False):
        super().__init__()
        if not mso.branch:
            raise ValueError("the embedding encoder requires OmniJev's branch path")
        self._mso = mso
        self.basis = basis
        self.source = dict(getattr(mso, "_superembedding_source", {}))
        self.backbone = mso.model.backbone
        self.backbone.requires_grad_(False)
        self.tune_lora = tune_lora
        if tune_lora:
            adapted = [p for name, p in self.backbone.named_parameters() if "lora_" in name]
            if not adapted:
                raise ValueError("no existing LoRA parameters found")
            for parameter in adapted:
                parameter.requires_grad_(True)
        self.backbone.eval()
        self.embedding_head = OptionEmbeddingHead.from_decision_head(
            mso.model.head, embed_dim=embed_dim, mode=mode, train_gate=train_gate,
        )

    def train(self, mode=True):
        super().train(mode)
        # Calling encoder.train() must not reactivate backbone LoRA dropout.
        if not self.tune_lora:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_features(self, state):
        """Return frozen (u, zq) for inexpensive repeated head training."""
        return self._branch_features(state, enable_grad=False)

    def _branch_features(self, state, enable_grad=False):
        from mso import branch as BR

        images = state.get("images")
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("state['images'] must be a nonempty list of image paths")
        video = state.get("video")
        if video and len(images) != 1:
            raise ValueError("video input requires exactly one frame-mosaic image")
        image_path = images[0]
        if len(images) > 1:
            from mso import panels
            from mso.infer import PANEL_DIR
            image_path = panels.compose_path(images, PANEL_DIR)

        mso = self._mso
        mso.dev = next(self.backbone.parameters()).device
        questions = {"attributes": {
            "type": "choice",
            "instructions": self.basis.instruction,
            "criteria": {a: None for a in self.basis.directions},
        }}
        # Do not call ask_branch(): its no_grad decorator covers the old head.
        # This no_grad scope ends before our new projection runs.
        with nullcontext() if enable_grad else torch.no_grad():
            keys_l, opts_l, ids_l, spans_l, pix = mso._encode_many(
                image_path, video, questions, ["attributes"],
            )
            if tuple(keys_l[0]) != self.basis.directions:
                raise RuntimeError("OmniJev changed the attribute order")
            opens, closes = spans_l[0]
            if len(opens) != len(self.basis.directions) or len(closes) != len(opens):
                raise RuntimeError("tokenized options do not match the attribute basis")
            # Same prefix boundary as ask_branch for one fixed question.
            prefix_length = max(1, opens[0] - 1)
            base = dict(pix)
            base["input_ids"] = torch.tensor(ids_l[0], device=mso.dev)[None]
            base["attention_mask"] = torch.ones_like(base["input_ids"])
            penc = BR.prefix_inputs(base, prefix_length)
            positions = BR.prefix_positions(mso.model, penc)
            rows = BR.rows_from_ids(
                ids_l[0], prefix_length, opens, closes, k=len(opts_l[0]),
            )
            [(u, zq, feats)] = BR.branch_questions(
                self.backbone, penc, [rows], mso.dev, mso.open_ids, mso.close_ids,
                lm_head=None, prefix_pos=positions, pad_id=mso.pad_id,
            )
            if feats is not None or u.shape[0] != len(self.basis.directions):
                raise RuntimeError("unexpected branch output")
            mso.last_input_tokens = prefix_length + sum(map(len, rows))
        return u, zq

    def forward(self, state):
        u, zq = self._branch_features(state, enable_grad=self.tune_lora)
        return self.embedding_head(u, zq)

    @torch.no_grad()
    def encode(self, state):
        previous = self.training
        self.eval()
        try:
            return self(state)
        finally:
            self.train(previous)

    def save_head(self, path):
        self.embedding_head.save(path, self.basis, self.source)

    def load_head(self, path, allow_source_relocation=False):
        head, _, source = OptionEmbeddingHead.load(path, expected_basis=self.basis)
        if head.input_dim != self.embedding_head.input_dim:
            raise ValueError("checkpoint backbone hidden size does not match")
        # References are recorded for provenance, not a substitute for weight hashes.
        if source.get('feature_backend') in {'omnijev_lora_disabled','base','flair'}:
            raise ValueError('Baseline head requires its matching feature extractor')
        if source and self.source and any(source.get(k)!=v for k,v in self.source.items()) and not allow_source_relocation:
            raise ValueError("checkpoint OmniJev source references do not match")
        self.embedding_head = head.to(next(self.backbone.parameters()).device)
        self.embedding_head.train(self.training)
