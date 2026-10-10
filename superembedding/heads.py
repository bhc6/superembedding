"""Gated and raw projection heads; no candidate probability features."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from .checkpoint import atomic_save


@dataclass(frozen=True)
class AttributeBasis:
    """The ordered directions and fixed instruction defining a memory bank."""

    directions: tuple[str, ...]
    instruction: str = (
        "Extract evidence about the object for the specified attribute direction."
    )

    def __post_init__(self):
        object.__setattr__(self, "directions", tuple(self.directions))
        if not self.directions or any(
            not isinstance(a, str) or not a.strip() for a in self.directions
        ):
            raise ValueError("directions must contain nonempty strings")
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("directions must be unique and ordered")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be nonempty")
        # OmniJev render_option truncates text at 200 characters.
        if any(len(a) > 200 for a in self.directions):
            raise ValueError("each direction must fit OmniJev's 200-character limit")
        if any(
            marker in text
            for text in (*self.directions, self.instruction)
            for marker in ("<|opt|>", "<|/opt|>")
        ):
            raise ValueError("attribute text must not contain option delimiters")


class EmbeddingOutput(NamedTuple):
    slots: torch.Tensor       # [..., K, d], normalized per slot
    embedding: torch.Tensor   # [..., K*d], normalized concatenation


class OptionEmbeddingHead(nn.Module):
    """P(opt(u) * tanh(que(zq))), with P(u) as a controlled alternative.

    Use from_decision_head() to initialize the gate from OmniJev, or load()
    to restore trained weights. The bare constructor initializes randomly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        embed_dim: int = 256,
        mode: str = "gated",
        train_gate: bool = False,
        num_slots: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()
        if mode not in {"gated", "raw", "residual"}:
            raise ValueError("mode must be gated, raw or residual")
        if any(not isinstance(d, int) or d <= 0 for d in (input_dim, hidden_dim, embed_dim)):
            raise ValueError("dimensions must be positive integers")
        if mode == "raw" and train_gate:
            raise ValueError("raw mode has no gate to train")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.mode = mode
        self.train_gate = train_gate
        self.num_slots, self.output_dim = num_slots, output_dim
        if output_dim is not None and (not num_slots or output_dim < 1):
            raise ValueError('output projection requires num_slots and positive output_dim')
        if mode in {"gated", "residual"}:
            self.opt = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.que = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.opt.requires_grad_(train_gate)
            self.que.requires_grad_(train_gate)
        width = hidden_dim if mode in {"gated", "residual"} else input_dim
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embed_dim))
        if mode == 'residual':
            self.residual_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, embed_dim))
        self.output_projection = nn.Linear(num_slots * embed_dim, output_dim, bias=False) if output_dim else nn.Identity()

    @classmethod
    def from_decision_head(cls, decision_head, embed_dim=256, mode="gated", train_gate=False, **kwargs):
        """Copy pretrained gate weights, without modifying the decision model."""
        input_dim = decision_head.opt[0].in_features
        hidden_dim = decision_head.opt[-1].out_features
        head = cls(input_dim, hidden_dim, embed_dim, mode, train_gate, **kwargs)
        if mode in {"gated", "residual"}:
            head.opt = deepcopy(decision_head.opt).float()
            head.que = deepcopy(decision_head.que).float()
            head.opt.requires_grad_(train_gate)
            head.que.requires_grad_(train_gate)
        return head.to(device=next(decision_head.parameters()).device, dtype=torch.float32)

    def forward(self, u: torch.Tensor, zq: torch.Tensor | None = None) -> EmbeddingOutput:
        if u.ndim < 2 or u.shape[-1] != self.input_dim or u.shape[-2] == 0:
            raise ValueError(f"u must have shape [..., K>0, {self.input_dim}]")
        dtype = self.projection[1].weight.dtype
        u = u.to(dtype=dtype)
        if self.num_slots is not None and u.shape[-2] != self.num_slots:
            raise ValueError('number/order of slots must match the trained output projection')
        if self.mode in {"gated", "residual"}:
            expected = u.shape[:-2] + (self.input_dim,)
            if zq is None or zq.shape != expected:
                raise ValueError(f"zq must have shape {tuple(expected)}")
            gate = torch.tanh(self.que(zq.to(dtype=dtype)))
            hidden = self.opt(u) * gate.unsqueeze(-2)
        else:
            hidden = u
        # Normalize in fp32 even if a caller enables autocast.
        projected = self.projection(hidden)
        if self.mode == 'residual':
            projected = projected + self.residual_projection(u)
        slots = F.normalize(projected.float(), dim=-1, eps=1e-8)
        embedding = F.normalize(self.output_projection(slots.flatten(start_dim=-2)).float(), dim=-1, eps=1e-8)
        return EmbeddingOutput(slots, embedding)

    def config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "mode": self.mode,
            "train_gate": self.train_gate,
            "num_slots": self.num_slots, "output_dim": self.output_dim,
        }

    def save(self, path, basis: AttributeBasis, source: dict[str, str] | None = None):
        """Save projection, gate, and basis. Does not save backbone weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save({
            "format_version": 1,
            "config": self.config(),
            "basis": asdict(basis),
            "source": dict(source or {}),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }, path)

    @classmethod
    def load(cls, path, expected_basis: AttributeBasis | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported embedding checkpoint format")
        basis = AttributeBasis(**payload["basis"])
        if expected_basis is not None and basis != expected_basis:
            raise ValueError("checkpoint attribute order or instruction does not match")
        head = cls(**payload["config"])
        head.load_state_dict(payload["state_dict"], strict=True)
        return head, basis, payload["source"]
