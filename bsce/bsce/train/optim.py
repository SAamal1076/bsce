"""TensorFlow-1 AdamOptimizer re-implemented for PyTorch.

tf.train.AdamOptimizer (used by every original training script) differs from
torch.optim.Adam in where epsilon enters:

    TF1  : lr_t = lr*sqrt(1-b2^t)/(1-b1^t);  w -= lr_t * m / (sqrt(v) + eps)
    torch: w -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)

``AdamTF1`` reproduces the TF1 update (verified against TF in tests/test_lamp_vs_tf.py).
The step counter is shared by all parameters of one optimizer, as TF's beta-power
accumulators are; a parameter whose gradient is None is skipped (TF filters None grads).

TF1 detail reproduced as well: Python-float hyper-parameters become float32 tensors
(lr, beta1, beta2, epsilon) and the beta-power accumulators are float32 variables, which
are then cast to the variable dtype inside ApplyAdam. Even float64 training therefore uses
float32-rounded hyper-parameters (``tf_float32_hparams=True``, default).
"""
from __future__ import annotations

import math

import numpy as np
import torch


class AdamTF1(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_scale=None,
                 tf_float32_hparams=True):
        if tf_float32_hparams:
            lr, beta1, beta2, epsilon = (float(np.float32(v)) for v in (lr, beta1, beta2, epsilon))
        self.f32 = tf_float32_hparams
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)
        self.t = 0
        self.b1_power = None   # running products, as TF's beta1_power / beta2_power variables
        self.b2_power = None
        # optional per-parameter gradient multipliers (ALISTA get_train_op)
        self.grad_scale = grad_scale or {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.t += 1
        g0 = self.param_groups[0]
        if self.b1_power is None:
            self.b1_power, self.b2_power = g0["beta1"], g0["beta2"]
        for group in self.param_groups:
            lr, b1, b2, eps = group["lr"], group["beta1"], group["beta2"], group["epsilon"]
            # Eigen ApplyAdam: alpha = lr*sqrt(1-beta2_power)/(1-beta1_power)
            lr_t = lr * math.sqrt(1 - self.b2_power) / (1 - self.b1_power)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                s = self.grad_scale.get(id(p))
                if s is not None:
                    g = g * s
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                m, v = st["m"], st["v"]
                # Eigen ApplyAdam order: m += (g-m)*(1-b1); v += (g^2-v)*(1-b2); var -= m*alpha/(sqrt(v)+eps)
                m.add_((g - m) * (1 - b1))
                v.add_((g * g - v) * (1 - b2))
                p.sub_((m * lr_t) / (torch.sqrt(v) + eps))
        # TF updates the beta powers after applying the update (float32 variables in TF1)
        if self.f32:
            self.b1_power = float(np.float32(self.b1_power) * np.float32(g0["beta1"]))
            self.b2_power = float(np.float32(self.b2_power) * np.float32(g0["beta2"]))
        else:
            self.b1_power *= g0["beta1"]
            self.b2_power *= g0["beta2"]
        return loss


    # the step counter and the beta powers are not in the per-parameter state: save them too (exact resume)
    def state_dict(self):
        d = super().state_dict()
        d["tf1"] = dict(t=self.t, b1_power=self.b1_power, b2_power=self.b2_power)
        return d

    def load_state_dict(self, state_dict):
        d = dict(state_dict)
        tf1 = d.pop("tf1", None)
        super().load_state_dict(d)
        if tf1 is not None:
            self.t, self.b1_power, self.b2_power = tf1["t"], tf1["b1_power"], tf1["b2_power"]


def make_optimizer(kind, params, lr, grad_scale=None):
    if kind == "adam_tf1":
        return AdamTF1(params, lr=lr, grad_scale=grad_scale)
    if kind == "adam_torch":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(kind)
