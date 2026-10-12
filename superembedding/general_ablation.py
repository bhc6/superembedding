"""Frozen general-feature heads; separate from the running LoRA experiment."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES=('shared_concat','output_equal','output_small','independent_equal',
               'independent_matched','mean_pool')


class GeneralAblationHead(nn.Module):
    def __init__(self, architecture, width=1024, slots=4, slot_dim=256):
        super().__init__()
        if architecture not in ARCHITECTURES:raise ValueError('Unknown architecture')
        self.architecture=architecture;self.slots=slots
        self.shared=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,slot_dim))
        self.output=nn.Identity()
        if architecture.startswith('independent'):
            d=slot_dim if architecture=='independent_equal' else slot_dim//slots
            if d<1:raise ValueError('Slot dimension too small')
            if architecture=='independent_equal':
                self.independent=nn.ModuleList([copy.deepcopy(self.shared) for _ in range(slots)])
            else:self.independent=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,d)) for _ in range(slots)])
            del self.shared
        elif architecture in {'output_equal','output_small'}:
            self.output=nn.Linear(slots*slot_dim,slots*slot_dim if architecture=='output_equal' else slot_dim,bias=False)
            if architecture=='output_equal':nn.init.eye_(self.output.weight)
        elif architecture=='mean_pool':
            self.shared=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,slots*slot_dim))

    def forward(self, u):
        if u.shape[-2]!=self.slots:raise ValueError('Wrong direction count')
        if self.architecture=='mean_pool':return F.normalize(self.shared(u.mean(-2)),dim=-1)
        if self.architecture.startswith('independent'):
            z=torch.stack([head(u[...,i,:]) for i,head in enumerate(self.independent)],dim=-2)
        else:z=self.shared(u)
        return F.normalize(self.output(F.normalize(z,dim=-1).flatten(-2)),dim=-1)
