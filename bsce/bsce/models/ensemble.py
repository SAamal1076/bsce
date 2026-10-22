"""Model-averaging ensemble (UTRR-E of Wu, Cheng, Eldar, Cioffi, IEEE TCOM 2025, Sec. IV-D): xhat = mean of the members'
outputs. The members are trained separately (UTRR: top-K = 0, 16, 64, 256); scripts/make_ensemble.py puts trained
networks together into one network file that evaluate.py / compare.py use like any other. Not trainable itself.
Saved as .pt (the members' configurations and weights); the .mat file holds the members' parameters with the prefix
m<i>_ for reference, and cannot be loaded back."""
from __future__ import annotations

from typing import List, Optional

import torch

from .common import UnrolledNet


class Ensemble(UnrolledNet):
    family = "ensemble"

    def __init__(self, A, members: List[dict], precision: str = "float32", device=None):
        super().__init__(A, precision)
        from . import build_model                  # late import: the registry imports this module
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.nets = torch.nn.ModuleList([build_model(dict(c), A) for c in members])
        self.T = max(int(c.get("T", 0)) for c in members)
        self.needs_noise_var = any(getattr(n, "needs_noise_var", False) for n in self.nets)
        self.cfg = dict(model="Ensemble", members=[dict(n.cfg) for n in self.nets], precision=precision)
        if device is not None:
            self.to(device)

    @classmethod
    def from_models(cls, models, precision: Optional[str] = None):
        m0 = models[0]
        e = cls(m0.A_np, [dict(m.cfg) for m in models], precision or m0.precision)
        for dst, src in zip(e.nets, models):
            dst.load_state_dict(src.state_dict())
        return e.to(m0.A.device)

    def stage_plan(self):
        return []

    def forward_stages(self, y, upto: Optional[int] = None, s2=None):
        outs = [n(y, s2=s2) if getattr(n, "needs_noise_var", False) else n(y) for n in self.nets]
        return [torch.stack(outs).mean(0)]

    def forward(self, y, s2=None):
        return self.forward_stages(y, s2=s2)[-1]

    def mat_dict(self):
        d = {}
        for i, n in enumerate(self.nets):
            d.update({f"m{i}_{k}": v for k, v in n.mat_dict().items()})
        return d

    def load_mat_dict(self, d):
        raise ValueError("an ensemble is loaded from its .pt file")
