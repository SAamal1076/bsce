"""Layer-wise (greedy + refinement) trainer, reproducing the original training scripts.

Two schedules:

* ``borgerding`` - setup_training/do_training of Borgerding et al., as adapted by
  Wei-Hu-Dai (GM-LAMP) and in the LADMM scripts. For every stage of
  ``model.stage_plan()``: train the new parameters alone at ``lr`` (if any), then all
  parameters existing so far at ``lr*fm`` for fm in ``refinements``. Validation every
  ``ivl`` iterations *before* the step; every ``check_mult*ivl`` iterations the stage stops
  if the best validation NMSE is older than ``better_wait`` iterations.
  ``warm_start=True`` reproduces GM-LAMP's do_training, which initialises a new layer's
  parameters with the trained values of the previous layer of the same kind.
  ``val_round=5`` reproduces ``nmse = round(nmse, 5)`` of the GM-LAMP / LADMM scripts.

* ``alista`` - utils/train.py of the ALISTA repository: per layer, train the new
  variables at ``lr``; then all trained variables at ``lr*d`` for d in ``lr_decay`` with
  per-variable gradient multipliers decayed by ``decay_rate`` after every layer;
  step *then* validate; stop check every ``10*ivl`` iterations.

Each stage uses a fresh optimizer (TF created one AdamOptimizer per stage).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .losses import LOSSES, nmse_loss, nmse_l2
from .optim import make_optimizer


@dataclass
class TrainConfig:
    schedule: str = "borgerding"           # 'borgerding' | 'alista'
    lr: float = 1e-3
    refinements: Sequence[float] = (0.5, 0.1, 0.01)
    lr_decay: Sequence[float] = (0.2, 0.02)  # alista
    decay_rate: float = 0.3                # alista
    maxit: int = 1_000_000
    better_wait: int = 5000
    ivl: int = 10
    check_mult: int = 100                  # borgerding 100, alista 10
    batch_size: int = 128
    loss: str = "nmse"                     # 'nmse' (GM-LAMP / LADMM) | 'l2' (Borgerding / ALISTA)
    val_metric: str = "nmse"               # 'nmse' | 'nmse_l2'
    val_round: Optional[int] = 5           # GM-LAMP/LADMM round(nmse, 5); None = no rounding
    warm_start: bool = False               # GM-LAMP: copy previous layer's trained values
    optimizer: str = "adam_tf1"            # 'adam_tf1' (exact TF1 Adam) | 'adam_torch'
    seed: int = 0
    val_chunk: Optional[int] = None        # evaluate validation set in chunks (memory)
    refine_name: str = "index"             # stage suffix: 'index' (GM-LAMP) | 'value' (Borgerding)
    epochs: int = 30                       # end2end schedule (THADMM-Net training loop)
    lr_list: Optional[Sequence[float]] = None   # end2end: one stage per learning rate, in this order; `epochs` is then
                                                # the budget of all stages together (UTRR: 0.005 ... 0.0001, 300)
    patience: Optional[int] = None         # end2end: stop a stage when the validation NMSE has not improved for this
                                           # many epochs (early stopping; UTRR: 10). None = run every epoch
    verbose: bool = True
    # --- options beyond the original code (default off = original behaviour) ---
    restore_best: bool = False             # at the end of every stage go back to the best validated weights
    val_groups: int = 1                    # >1: validation set = val_groups equal SNR groups; the
                                           #     validation value is the mean over groups in dB
    val_mean_db: bool = False              # train.py sets val_groups = number of validation SNRs
    # --- noise / step schedule options (14 Oct 2026) ---
    noise: str = "file"                    # 'file': y from the measurement files (original);
                                           # 'online': fresh noise every minibatch, one SNR per sample ~ U(snr_range)
    snr_range: Optional[Sequence[float]] = None   # online noise range (train.py sets it from the band)
    snr_dist: str = "uniform"                     # online noise SNRs of a minibatch: 'uniform' = i.i.d. U(snr_range) (as
                                                  # before); 'fair' (v26) = spread evenly over the range in every batch
                                                  # (one SNR in each of B equal parts, random order)
    snr_step: Optional[float] = None              # with 'fair': the grid lo, lo+step, ..., hi (equal counts per batch)
    steps: int = 20000                     # schedule 'steps': end-to-end Adam steps
    lr_final_frac: float = 0.05            # schedule 'steps': cosine decay from lr to lr*lr_final_frac
    grad_clip: Optional[float] = None      # clip the global gradient norm (None = off)
    val_every: int = 500                   # schedule 'steps': validation interval (best state kept if restore_best)
    val_round_mode: str = "model"          # 'model': np.round in the network precision (TF scripts);
                                           # 'python': round(float, n) as untitled2.py (nmse.item() then round)
    # --- resume inside a stage (16 Oct 2026) ---
    checkpoint_minutes: float = 10.0       # with a checkpoint path: also save INSIDE a stage every this many minutes
                                           # (weights, optimizer, random streams, best state, iteration), so an
                                           # interrupted run continues where it stopped, exactly; 0 = only after
                                           # every stage (as before)
    checkpoint_iters: Optional[int] = None  # also every this many iterations / steps / epochs (tests)
    # --- early stopping (16 Oct 2026, v25) ---
    patience_steps: Optional[int] = 5000   # schedule 'steps': stop when the best validation has not improved by at least
                                           # min_gain_db during the last this many steps (best state kept as before);
                                           # 0 / None = always all `steps`
    min_gain_db: Optional[float] = None    # the gain (dB) that counts as an improvement for the stopping rules:
                                           # None = 0.01 dB for 'steps', 0 = any gain (the original rule) for the
                                           # layer-wise schedules (better_wait) and 'end2end' (patience)

    # --- combiner (v29) ---
    combiner: str = "fixed"                # 'fixed': the scenario's combiner A (as before); 'random': a new random
                                           # +-1/sqrt(M) combiner every combiner_every minibatches (online noise only),
                                           # for networks with set_combiner (analytic LADMM, Graph-LADMM analytic):
                                           # they then work with any combiner. Validation, checkpoints and the saved
                                           # network use the scenario's A.
                                           # 'mix' (v31): every draw takes an M from combiner_M and a family from
                                           # combiner_types, so the network also sees other sizes and structured
                                           # combiners in training (networks with set_combiner; any M from v31)
    combiner_every: int = 1
    gm_symmetry: str = "break"             # v31.1: 'break' = identical GM-LAMP mixture components get the 'spread' offset
                                           # before a stage trains them (else they never separate in PyTorch and the
                                           # denoiser stays one Gaussian); 'keep' = the original arithmetic (TF tests)
    combiner_M: Optional[List[int]] = None     # 'mix': the numbers of RF chains to draw from (None = the scenario's M)
    combiner_types: Optional[List[str]] = None  # 'mix': from pm1 (random +-1), gauss, sparse (16 beams per RF chain),
                                           # dct (random rows of the orthonormal DCT-II); None = pm1

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def last_gain_index(vals_db, min_gain: float) -> int:
    """Index of the last validation that improved on the reference by at least min_gain dB; the reference is the value
    of the previous such validation (the first value to start with). min_gain 0: the index of the best value (first)."""
    ref, j = vals_db[0], 0
    for k, v in enumerate(vals_db):
        if v < ref - min_gain:
            ref, j = v, k
    return j


def build_stages(model, cfg: TrainConfig) -> List[Dict]:
    """Expand model.stage_plan() into the list of training sub-stages of the original code."""
    plan = model.stage_plan()
    stages = []
    if cfg.schedule == "borgerding":
        for st in plan:
            if st.new:
                stages.append(dict(name=st.name, out=st.index, params=list(st.new), lr=cfg.lr,
                                   kind=st.kind, first=True, scale={}))
            for i, fm in enumerate(cfg.refinements):
                suffix = str(i) if cfg.refine_name == "index" else str(fm)
                stages.append(dict(name=f"{st.name} trainrate={suffix}", out=st.index, params=list(st.all),
                                   lr=fm * cfg.lr, kind=st.kind, first=False, scale={}))
    elif cfg.schedule == "alista":
        mult = {n: 1.0 for n in model.param_names()}
        trained: List[str] = []
        for st in plan:
            if st.kind != "layer":
                continue
            new = [n for n in st.new if n not in trained]
            stages.append(dict(name=st.name, out=st.index, params=new, lr=cfg.lr, kind=st.kind,
                               first=True, scale={}))
            trained += new
            for d in cfg.lr_decay:
                stages.append(dict(name=f"{st.name} lr={cfg.lr * d}", out=st.index, params=list(trained),
                                   lr=cfg.lr * d, kind=st.kind, first=False,
                                   scale={n: mult[n] for n in trained}))
            for n in trained:
                mult[n] *= cfg.decay_rate
    elif cfg.schedule == "end2end" and cfg.lr_list:
        last = plan[-1]
        for lr in cfg.lr_list:
            stages.append(dict(name=f"end2end lr={float(lr):g}", out=last.index, params=list(model.param_names()),
                               lr=float(lr), kind="all", first=False, scale={}))
    elif cfg.schedule in ("end2end", "steps"):
        last = plan[-1]
        stages.append(dict(name=cfg.schedule, out=last.index, params=list(model.param_names()), lr=cfg.lr,
                           kind="all", first=False, scale={}))
    else:
        raise ValueError(cfg.schedule)
    return stages


COMBINER_TYPES = ("pm1", "gauss", "sparse", "dct")
_DCT = {}


def make_train_combiner(kind: str, M: int, N: int, gen: torch.Generator) -> np.ndarray:
    """One training combiner (M x N, float64) of a family (train.combiner=mix, v31), drawn with gen:
    pm1 random +-1/sqrt(M); gauss N(0, 1)/sqrt(M); sparse 16 random beams per RF chain, +-1/4 (partially connected);
    dct M random rows of the orthonormal DCT-II. The same families as check_combiners.py's test combiners (other draws)."""
    if kind == "pm1":
        return (((torch.rand((M, N), generator=gen, dtype=torch.float64) > 0.5).double() * 2 - 1) / math.sqrt(M)).numpy()
    if kind == "gauss":
        return (torch.randn((M, N), generator=gen, dtype=torch.float64) / math.sqrt(M)).numpy()
    if kind == "sparse":
        k = min(16, N)
        S = torch.zeros((M, N), dtype=torch.float64)
        for m in range(M):
            idx = torch.randperm(N, generator=gen)[:k]
            S[m, idx] = (torch.rand(k, generator=gen, dtype=torch.float64) > 0.5).double() * 2 - 1
        return (S / math.sqrt(k)).numpy()
    if kind == "dct":
        if N not in _DCT:
            from scipy.fft import dct
            _DCT[N] = dct(np.eye(N), norm="ortho", axis=0)
        rows = torch.sort(torch.randperm(N, generator=gen)[:M]).values.numpy()
        return _DCT[N][rows].copy()
    raise ValueError(f"combiner family {kind!r}: one of {COMBINER_TYPES}")

class Trainer:
    def __init__(self, model, cfg: TrainConfig, x_train, y_train, x_val, y_val,
                 checkpoint: Optional[str] = None, export_fn=None, sampler=None, stage_callback=None, layer_callback=None,
                 s2_train=None, s2_val=None, noise_model=None):
        """y_train may be None with cfg.noise == 'online'. s2_train / s2_val: noise variance of every column
        (needed by models with needs_noise_var, e.g. NF-LADMM; with online noise s2_train is drawn per batch).
        noise_model: 'colored' (default, y = A (x + n)) or 'white' (bsce/data/noise.py): the noise of the online
        minibatches (measurement files and the validation set carry their own)."""
        self.model, self.cfg = model, cfg
        from ..data import noise as noise_models
        self.noise_model = noise_models.canonical(noise_model)
        self.Twhite = (torch.as_tensor(noise_models.whitening_map(model.A_np)).to(model.A.device, model.rdtype)
                       if self.noise_model == "white" else None)
        dev = model.A.device
        cd = model.cdtype
        self.xt = torch.as_tensor(np.asarray(x_train) if not torch.is_tensor(x_train) else x_train).to(dev, cd)
        if cfg.noise == "online":
            if not cfg.snr_range:
                raise ValueError("noise='online' needs train.snr_range = [low_dB, high_dB]")
            self.yt = None
        else:
            self.yt = torch.as_tensor(np.asarray(y_train) if not torch.is_tensor(y_train) else y_train).to(dev, cd)
        rd = model.rdtype
        self.st = None if s2_train is None else torch.as_tensor(np.asarray(s2_train, float).ravel()).to(dev, rd)
        self.sv = None if s2_val is None else torch.as_tensor(np.asarray(s2_val, float).ravel()).to(dev, rd)
        if getattr(model, "needs_noise_var", False):
            if self.sv is None:
                raise ValueError(f"{type(model).__name__} needs the noise variance of the validation columns (s2_val)")
            if cfg.noise != "online" and self.st is None:
                raise ValueError(f"{type(model).__name__} needs the noise variance of the training columns: use "
                                 "train.noise=online, or measurement files with 'snr_db' (scripts/make_sv_measurements.py)")
        self.xv = torch.as_tensor(np.asarray(x_val) if not torch.is_tensor(x_val) else x_val).to(dev, cd)
        self.yv = torch.as_tensor(np.asarray(y_val) if not torch.is_tensor(y_val) else y_val).to(dev, cd)
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        # sampler() -> index array of one minibatch (default: uniform with replacement, as
        # np.random.choice(trainingsize, size=batch) in the originals). Tests replay the
        # exact indices drawn by the original scripts.
        self.sampler = sampler
        self.stage_callback = stage_callback   # f(stage_dict, model) after every stage
        self.layer_callback = layer_callback   # f(out_index, model, val_nmse_db) when all stages of an output index are done
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.export_fn = export_fn
        self.state = {"done": [], "log": [], "layer_nmse_db": []}
        self._last_ck = time.time()
        self._resume = None                    # the saved position inside the current stage (load_checkpoint)
        self.loss_fn = LOSSES[cfg.loss]
        self.val_fn = nmse_loss if cfg.val_metric == "nmse" else nmse_l2
        self._nb = 0
        self.A0 = None
        if cfg.gm_symmetry not in ("break", "keep"):
            raise ValueError(f"train.gm_symmetry: 'break' or 'keep' (got {cfg.gm_symmetry!r})")
        if (cfg.combiner or "fixed") != "fixed":
            if cfg.combiner not in ("random", "mix"):
                raise ValueError(f"train.combiner: 'fixed', 'random' or 'mix' (got {cfg.combiner!r})")
            bad = [t for t in (cfg.combiner_types or []) if t not in COMBINER_TYPES]
            if bad:
                raise ValueError(f"train.combiner_types: {bad} not in {COMBINER_TYPES}")
            if any(not 0 < int(m) <= model.N for m in (cfg.combiner_M or [])):
                raise ValueError(f"train.combiner_M: every M must be 1 .. N = {model.N} (got {cfg.combiner_M})")
            if cfg.combiner == "random" and (cfg.combiner_M or cfg.combiner_types):
                raise ValueError("train.combiner_M / combiner_types need train.combiner=mix")
            if cfg.noise != "online":
                raise ValueError("train.combiner=random needs train.noise=online (the measurements depend on the combiner)")
            if self.noise_model == "white":
                raise ValueError("train.combiner=random is for the colored noise y = A (x + n)")
            if not hasattr(model, "set_combiner"):
                raise ValueError(f"{type(model).__name__} cannot change its combiner (train.combiner=random)")
            self.A0 = np.array(model.A_np, copy=True)
            model.set_combiner(self.A0)              # raises for networks learned for one combiner

    # ------------------------------------------------------------------------------
    def _fwd(self, y, upto, s2=None):
        if getattr(self.model, "needs_noise_var", False):
            return self.model.forward_stages(y, upto=upto, s2=s2)
        return self.model.forward_stages(y, upto=upto)

    def _sv(self, s, e):
        return None if self.sv is None else self.sv[s:e]

    @torch.no_grad()
    def _val_xhat(self, out_index: int):
        if not self.cfg.val_chunk:
            return self._fwd(self.yv, out_index, self.sv)[-1]
        K, c = self.xv.shape[1], self.cfg.val_chunk
        return torch.cat([self._fwd(self.yv[:, s:s + c], out_index, self._sv(s, s + c))[-1] for s in range(0, K, c)], 1)

    def _home_combiner(self):
        if self.A0 is not None:
            self.model.set_combiner(self.A0)

    @torch.no_grad()
    def validate(self, out_index: int) -> float:
        self._home_combiner()
        g = int(self.cfg.val_groups or 1)
        if g > 1:
            # one value per validation SNR group, averaged in dB (geometric mean of the linear NMSE)
            xh = self._val_xhat(out_index)
            K = self.xv.shape[1]
            if K % g:
                raise ValueError(f"validation size {K} is not divisible by val_groups={g}")
            w = K // g
            vs = [float(self.val_fn(xh[:, i * w:(i + 1) * w], self.xv[:, i * w:(i + 1) * w])) for i in range(g)]
            v = float(np.exp(np.mean(np.log(np.maximum(vs, 1e-300)))))
        elif self.cfg.val_chunk:
            # NMSE over the whole set computed from chunked sums
            num = den = 0.0
            K = self.xv.shape[1]
            for s in range(0, K, self.cfg.val_chunk):
                xh = self._fwd(self.yv[:, s:s + self.cfg.val_chunk], out_index, self._sv(s, s + self.cfg.val_chunk))[-1]
                xs = self.xv[:, s:s + self.cfg.val_chunk]
                num += float(torch.sum(torch.square(torch.abs(xh - xs))))
                den += float(torch.sum(torch.square(torch.abs(xs))))
            v = num / den
        else:
            xh = self._fwd(self.yv, out_index, self.sv)[-1]
            v = float(self.val_fn(xh, self.xv))
        if self.cfg.val_round is not None:
            # np.round in the network's precision, as round(nmse, 5) / np.round(nmse, 5) on the
            # numpy scalar returned by sess.run (float64 for GM-LAMP, float32 for LADMM_train.py)
            if self.cfg.val_round_mode == "python":
                v = round(float(v), self.cfg.val_round)
            else:
                rd = np.float32 if self.model.rdtype == torch.float32 else np.float64
                v = float(np.round(rd(v), self.cfg.val_round))
        return v

    def _batch(self):
        """-> (y, x, s2) of one minibatch (s2 None when unknown / not needed)."""
        if self.sampler is not None:
            idx = torch.as_tensor(np.asarray(self.sampler()), dtype=torch.long)
        else:
            idx = torch.randint(0, self.xt.shape[1], (self.cfg.batch_size,), generator=self.gen)
        idx = idx.to(self.xt.device)
        x = self.xt[:, idx]
        if self.cfg.noise == "online":
            lo, hi = map(float, self.cfg.snr_range)
            B = x.shape[1]
            snr = self._batch_snr(lo, hi, B)
            s2 = 10.0 ** (-snr / 10)
            sd = torch.sqrt(s2 / 2).to(x.device, self.model.rdtype)
            n = torch.complex(torch.randn(x.shape, generator=self.gen, dtype=torch.float64),
                              torch.randn(x.shape, generator=self.gen, dtype=torch.float64)).to(x.device, x.dtype)
            if self.A0 is not None and self._nb % max(1, int(self.cfg.combiner_every)) == 0:
                M, N = self.A0.shape
                if self.cfg.combiner == "mix":
                    self.model.set_combiner(self._draw_combiner(N))
                else:
                    Ab = ((torch.rand((M, N), generator=self.gen, dtype=torch.float64) > 0.5).double() * 2 - 1) / math.sqrt(M)
                    self.model.set_combiner(Ab.numpy())
            self._nb += 1
            A = self.model.A
            xn = x + sd * n
            y = torch.complex(A @ xn.real, A @ xn.imag)
            if self.Twhite is not None:                     # white: the same draw whitened, y = A x + T A n
                sx = torch.complex(A @ x.real, A @ x.imag)
                d = y - sx
                y = sx + torch.complex(self.Twhite @ d.real, self.Twhite @ d.imag)
            return y, x, s2.to(x.device, self.model.rdtype)
        return self.yt[:, idx], x, (None if self.st is None else self.st[idx])

    def _draw_combiner(self, N: int) -> np.ndarray:
        """train.combiner=mix (v31): one combiner, M from combiner_M, family from combiner_types, drawn with the
        training generator (so a resumed run draws the same ones). Redrawn in the rare case of dependent rows."""
        Ms = [int(m) for m in (self.cfg.combiner_M or [self.A0.shape[0]])]
        types = list(self.cfg.combiner_types or ["pm1"])
        for _ in range(20):
            M = Ms[int(torch.randint(0, len(Ms), (1,), generator=self.gen))]
            kind = types[int(torch.randint(0, len(types), (1,), generator=self.gen))]
            A = make_train_combiner(kind, M, N, self.gen)
            if np.linalg.matrix_rank(A) == M:
                return A
        raise RuntimeError(f"train.combiner=mix: no full-rank {kind} combiner of size {M} x {N} in 20 draws")

    def _batch_snr(self, lo: float, hi: float, B: int) -> torch.Tensor:
        """SNRs (dB, float64) of one online minibatch: i.i.d. U(lo, hi), or 'fair' (cfg.snr_dist): one SNR in each of B
        equal parts of [lo, hi] (cfg.snr_step: the grid lo, lo+step, ..., hi, cycled from a random start), shuffled."""
        if self.cfg.snr_dist in (None, "uniform"):
            return lo + (hi - lo) * torch.rand(B, generator=self.gen, dtype=torch.float64)
        if self.cfg.snr_dist != "fair":
            raise ValueError(f"train.snr_dist: 'uniform' or 'fair' (got {self.cfg.snr_dist!r})")
        perm = torch.randperm(B, generator=self.gen)
        if self.cfg.snr_step:
            st = float(self.cfg.snr_step)
            n = int(math.floor((hi - lo) / st + 1e-9)) + 1
            j0 = int(torch.randint(0, n, (1,), generator=self.gen))
            return lo + st * ((torch.arange(B, dtype=torch.float64) + j0) % n)[perm]
        u = torch.rand(B, generator=self.gen, dtype=torch.float64)
        return lo + (hi - lo) * (perm.to(torch.float64) + u) / B

    def train_step(self, stage, opt, params) -> float:
        y, x, s2 = self._batch()
        opt.zero_grad(set_to_none=True)
        xh = self._fwd(y, stage["out"], s2)[-1]
        loss = self.loss_fn(xh, x)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        opt.step()
        return float(loss.detach())

    # ------------------------------------------------------------------------------
    def _warm_start(self, stage, last_new):
        """GM-LAMP do_training: assign new vars from the previous same-kind var_list."""
        prev = last_new.get(stage["kind"])
        if prev and len(prev) == len(stage["params"]):
            with torch.no_grad():
                for dst, src in zip(stage["params"], prev):
                    self.model.P(dst).copy_(self.model.P(src))
        last_new[stage["kind"]] = list(stage["params"])

    def run(self, stages: Optional[List[Dict]] = None) -> Dict:
        cfg = self.cfg
        stages = stages if stages is not None else build_stages(self.model, cfg)
        if self.checkpoint and self._ck_path().exists():
            fresh = self._fresh_copy() if cfg.gm_symmetry == "break" and hasattr(self.model, "identical_gm_components") \
                else None
            self.load_checkpoint()
            if fresh is not None:
                self._drop_pre_v31_1_checkpoint(stages, fresh)
        last_new: Dict[str, List[str]] = {}
        nmse_db = None
        for st in stages:
            part = self._resume if self._resume is not None and self._resume.get("stage") == st["name"] else None
            if st["first"] and cfg.warm_start and st["params"]:
                if st["name"] not in self.state["done"] and part is None:
                    self._warm_start(st, last_new)
                else:                                  # done, or resumed inside it (its weights are restored)
                    last_new[st["kind"]] = list(st["params"])
            if st["name"] in self.state["done"]:
                continue
            if not st["params"]:
                continue
            if part is None and cfg.gm_symmetry == "break" and hasattr(self.model, "break_gm_symmetry"):
                hit = self.model.break_gm_symmetry(st["params"])      # v31.1: see LAMP.break_gm_symmetry
                if hit:
                    self.state.setdefault("gm_symmetry_broken", []).extend(hit)
                    if cfg.verbose:
                        print(f"[bsce] {', '.join(hit)}: the {self.model.K} mixture components were identical (GM-LAMP's "
                              "original init); in PyTorch they would never separate and the denoiser would stay one "
                              "Gaussian. "
                              "Offset as gm_init=spread (train.gm_symmetry=keep: unchanged)")
            params = [self.model.P(n) for n in st["params"]]
            scale = {id(self.model.P(n)): s for n, s in st["scale"].items()} if st["scale"] else None
            opt = make_optimizer(cfg.optimizer, params, st["lr"], grad_scale=scale)
            if part is not None:
                try:
                    opt.load_state_dict(part["opt"])
                except ValueError:                     # v31: a pre-v31 Graph-LADMM checkpoint has 2 more parameters
                    print(f"[bsce] stage '{st['name']}': the optimizer state of this checkpoint does not fit the "
                          "network (written before v31, which left out the parameters that never reach the output); "
                          "the weights and the position are resumed, the optimizer state starts again")
                self._resume = None
                if cfg.verbose:
                    print(f"[bsce] resumed inside stage '{st['name']}' at {part['where']}")
            if cfg.verbose:
                print(f"{st['name']}  [{'extending ' if st['first'] else 'fine tuning '}{','.join(st['params'][-6:])}] lr={st['lr']:.2e}")
            if cfg.schedule in ("end2end", "steps"):
                nmse_db = (self._run_end2end(st, opt, part) if cfg.schedule == "end2end" else
                           self._run_steps(st, opt, params, part))
                self.state["done"].append(st["name"])
                self.state["layer_nmse_db"].append(nmse_db)
                self.state.pop("partial", None)
                if self.checkpoint:
                    self.save_checkpoint()
                continue
            hist, t0, i = [], time.time(), 0
            best_state = None
            i0 = 0
            if part is not None:
                i0, hist, best_state = part["i"], list(part["hist"]), part["best_state"]
            for i in range(i0, cfg.maxit + 1):
                if i > i0 and self._due(i):
                    self._save_partial(st, opt, f"iteration {i}", i=i, hist=list(hist), best_state=best_state)
                if cfg.schedule == "alista":
                    self.train_step(st, opt, params)
                if i % cfg.ivl == 0:
                    v = self.validate(st["out"])
                    if math.isnan(v):
                        raise RuntimeError("nmse is NaN")
                    if cfg.restore_best and (not hist or v < min(hist)):
                        best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
                    hist.append(v)
                    nmse_db = 10 * math.log10(max(v, 1e-300))
                    if i % (cfg.check_mult * cfg.ivl) == 0:
                        best = 10 * math.log10(max(min(hist), 1e-300))
                        if cfg.verbose:
                            print(f"  i={i:<7d} nmse={nmse_db:.4f} dB (best={best:.4f})  {time.time() - t0:.1f}s")
                        g = self._min_gain()
                        if g > 0:                    # only a gain >= g resets the wait (min_gain_db)
                            age = len(hist) - last_gain_index([10 * math.log10(max(h, 1e-300)) for h in hist], g) - 1
                        else:                        # the original rule: any new best resets it
                            age = len(hist) - int(np.argmin(hist)) - 1
                        if age * cfg.ivl > cfg.better_wait:
                            break
                if cfg.schedule != "alista":
                    self.train_step(st, opt, params)
            if cfg.restore_best and best_state is not None:
                self.model.load_state_dict(best_state)
                nmse_db = 10 * math.log10(max(min(hist), 1e-300))
                if cfg.verbose:
                    print(f"  restored best weights: nmse={nmse_db:.4f} dB (iteration {int(np.argmin(hist)) * cfg.ivl})")
            self.state["done"].append(st["name"])
            self.state.pop("partial", None)
            self.state["log"].append(dict(stage=st["name"], nmse_db=nmse_db, iters=i, nmse_hist=hist[-3:]))
            if not any(s["out"] == st["out"] and s["name"] not in self.state["done"] for s in stages):
                self.state["layer_nmse_db"].append(nmse_db)
                if self.layer_callback is not None:
                    self.layer_callback(st["out"], self.model, nmse_db)
            if self.checkpoint:
                self.save_checkpoint()
            if self.stage_callback is not None:
                self.stage_callback(st, self.model)
        self._home_combiner()
        return self.state

    def _run_end2end(self, st, opt, part=None) -> float:
        """THADMM-Net training loop: shuffled minibatches per epoch, validation per epoch.
        part: the position saved inside this stage (resume after the last saved epoch)."""
        cfg = self.cfg
        K = self.xt.shape[1]
        best, nmse_db, best_state = float("inf"), None, None
        stale, ep0 = 0, 0
        g = self._min_gain()
        if part is not None:
            best, nmse_db, best_state, stale, ep0 = part["best"], part["nmse_db"], part["best_state"], part["stale"], part["ep"]
        ref = part.get("ref", best) if part is not None else best     # the value of the last gain >= g
        shared = bool(cfg.lr_list)                   # lr_list: one epoch budget for all the stages together
        keep = part.get("keep", best) if part is not None else best   # validation NMSE of best_state
        if part is None and cfg.restore_best and shared and self.state.get("epochs_done", 0):
            # a later lr_list stage starts from the previous stage's best weights: keep them unless beaten
            keep = self.validate(st["out"])
            best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
        n_ep = cfg.epochs - int(self.state.get("epochs_done", 0)) if shared else cfg.epochs - ep0
        for ep in range(ep0, ep0 + max(n_ep, 0)):
            if cfg.patience and stale >= cfg.patience:
                if cfg.verbose:
                    print(f"  early stop: no improvement for {cfg.patience} epochs")
                break
            perm = torch.randperm(K, generator=self.gen).to(self.xt.device)
            for s in range(0, K, cfg.batch_size):
                idx = perm[s:s + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                if self.yt is None:
                    raise ValueError("schedule 'end2end' needs measurement files; use schedule 'steps' with online noise")
                loss = self.loss_fn(self._fwd(self.yt[:, idx], st["out"], None if self.st is None else self.st[idx])[-1],
                                    self.xt[:, idx])
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]],
                                                   cfg.grad_clip)
                opt.step()
            v = self.validate(st["out"])
            if math.isnan(v):
                raise RuntimeError("nmse is NaN")
            nmse_db = 10 * math.log10(max(v, 1e-300))
            if cfg.restore_best and v < keep:
                keep = v
                best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
            if g > 0:                                # min_gain_db: only a gain >= g resets the patience
                if nmse_db < 10 * math.log10(max(ref, 1e-300)) - g:
                    stale, ref = 0, v
                else:
                    stale += 1
            else:                                    # the original rule: any new best resets it
                stale = 0 if v < best else stale + 1
            best = min(best, v)
            if shared:
                self.state["epochs_done"] = int(self.state.get("epochs_done", 0)) + 1
            self.state["log"].append(dict(stage=f"{st['name']} epoch {ep + 1}" if shared else f"epoch {ep + 1}",
                                          nmse_db=nmse_db))
            if cfg.verbose:
                print(f"  epoch {ep + 1:3d}  val nmse={nmse_db:.4f} dB (best={10 * math.log10(best):.4f})")
            if self._due(ep + 1):
                self._save_partial(st, opt, f"epoch {ep + 1}", ep=ep + 1, best=best, nmse_db=nmse_db,
                                   best_state=best_state, stale=stale, ref=ref, keep=keep)
        if cfg.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
            nmse_db = 10 * math.log10(max(keep, 1e-300))
        return nmse_db

    def _run_steps(self, st, opt, params, part=None) -> float:
        """End-to-end training for cfg.steps minibatches: Adam with cosine decay lr -> lr*lr_final_frac,
        optional gradient clipping, validation every cfg.val_every steps (best state kept if restore_best).
        part: the position saved inside this stage (resume at that step)."""
        cfg = self.cfg
        lr0 = st["lr"]
        best, best_state, nmse_db, t0 = float("inf"), None, None, time.time()
        hist, i0, stopped = [], 0, None
        if part is not None:
            i0, best, best_state, nmse_db, hist = part["i"], part["best"], part["best_state"], part["nmse_db"], list(part["hist"])
        for i in range(i0, cfg.steps + 1):
            if i > i0 and i < cfg.steps and self._due(i):
                self._save_partial(st, opt, f"step {i}", i=i, best=best, best_state=best_state, nmse_db=nmse_db,
                                   hist=list(hist))
            if i % cfg.val_every == 0 or i == cfg.steps:
                v = self.validate(st["out"])
                if math.isnan(v):
                    raise RuntimeError("nmse is NaN")
                nmse_db = 10 * math.log10(max(v, 1e-300))
                hist.append((i, nmse_db))
                if v < best:
                    best = v
                    if cfg.restore_best:
                        best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
                if cfg.verbose:
                    print(f"  step={i:<7d} val nmse={nmse_db:.4f} dB (best={10 * math.log10(best):.4f})  "
                          f"lr={opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.1f}s")
                if cfg.patience_steps and 0 < i < cfg.steps:
                    j = hist[last_gain_index([h[1] for h in hist], self._min_gain())][0]
                    if i - j >= int(cfg.patience_steps):
                        stopped = dict(step=i, last_gain_step=j, patience_steps=int(cfg.patience_steps),
                                       min_gain_db=self._min_gain())
                        if cfg.verbose:
                            print(f"  early stop at step {i}: no gain of {self._min_gain():g} dB or more since step {j} "
                                  f"(patience {int(cfg.patience_steps)} steps; train.patience_steps=0 runs all "
                                  f"{cfg.steps} steps)")
                        break
            if i == cfg.steps:
                break
            frac = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * i / cfg.steps))
            for g in opt.param_groups:
                g["lr"] = lr0 * frac
            self.train_step(st, opt, params)
        if cfg.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
            nmse_db = 10 * math.log10(max(best, 1e-300))
            if cfg.verbose:
                print(f"  restored best weights: nmse={nmse_db:.4f} dB")
        self.state["log"].append(dict(stage=st["name"], nmse_db=nmse_db, iters=stopped["step"] if stopped else cfg.steps,
                                      val_hist=hist, **({"early_stop": stopped} if stopped else {})))
        return nmse_db

    # ------------------------------------------------------------------------------
    def _min_gain(self) -> float:
        """The gain (dB) that counts as an improvement: train.min_gain_db, else 0.01 dB for 'steps' and 0 (any gain,
        the original rule) for the layer-wise schedules and 'end2end'."""
        if self.cfg.min_gain_db is not None:
            return float(self.cfg.min_gain_db)
        return 0.01 if self.cfg.schedule == "steps" else 0.0

    def _due(self, k: int) -> bool:
        """Time to save inside the stage? (every cfg.checkpoint_minutes, or every cfg.checkpoint_iters)."""
        if not self.checkpoint:
            return False
        if self.cfg.checkpoint_iters and k % int(self.cfg.checkpoint_iters) == 0:
            return True
        m = self.cfg.checkpoint_minutes
        return bool(m) and m > 0 and time.time() - self._last_ck >= 60.0 * m

    def _save_partial(self, st, opt, where: str, **pos):
        """Checkpoint inside stage st: its position (iteration, history, best state), the optimizer, and (in
        save_checkpoint) the weights and random streams. The next run continues from here exactly."""
        best = pos.get("best_state")
        if best is not None:
            pos["best_state"] = {k: v.detach().cpu().clone() for k, v in best.items()}
        self.state["partial"] = dict(stage=st["name"], where=where, opt=opt.state_dict(), **pos)
        self.save_checkpoint()

    def _ck_path(self) -> Path:
        """<checkpoint>.pt (not with_suffix: a run name may contain a dot, e.g. an override lr=0.0005)."""
        return self.checkpoint.parent / (self.checkpoint.name + ".pt")

    def save_checkpoint(self):
        self._home_combiner()
        p = self._ck_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        # the random streams are saved too: a resumed run then continues exactly as an uninterrupted one
        # (minibatches / online noise come from self.gen; the global torch and numpy states for anything else)
        torch.save({"model_state": self.model.state_dict(), "cfg_model": self.model.cfg,
                    "train_state": self.state, "train_cfg": asdict(self.cfg),
                    "rng": {"gen": self.gen.get_state(), "torch": torch.get_rng_state(), "numpy": np.random.get_state()}},
                   str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)                 # never a half-written checkpoint (power cut during the save)
        self._last_ck = time.time()
        if self.export_fn is not None:
            self.export_fn(self.model, self.checkpoint, self.state)

    def _fresh_copy(self):
        """The state before a checkpoint is loaded: the new network and the random streams (for a fresh start)."""
        return dict(model={k: v.detach().clone() for k, v in self.model.state_dict().items()},
                    gen=self.gen.get_state(), torch=torch.get_rng_state(), numpy=np.random.get_state())

    def _drop_pre_v31_1_checkpoint(self, stages, fresh):
        """v31.1: a GM-LAMP checkpoint written before v31.1 trained its mixtures with identical components (a linear
        denoiser). Resuming it would continue that training, so it is set aside and the run starts again."""
        done = set(self.state.get("done", []))
        trained = [n for st in stages if st["name"] in done or (self._resume or {}).get("stage") == st["name"]
                   for n in st["params"]]
        same = self.model.identical_gm_components(sorted(set(trained)))
        if not same:
            return
        old = self._ck_path()
        keep = old.parent / (old.name[:-3] + ".pre_v31_1.pt")
        os.replace(old, keep)
        self.model.load_state_dict(fresh["model"])
        self.gen.set_state(fresh["gen"])
        torch.set_rng_state(fresh["torch"])
        np.random.set_state(fresh["numpy"])
        self.state = {"done": [], "log": [], "layer_nmse_db": []}
        self._resume = None
        print(f"[bsce] the checkpoint was written before v31.1: {', '.join(same)} trained with identical mixture "
              f"components (one Gaussian, no sparsity). It is not resumed; the training starts again (the old checkpoint is "
              f"kept as {keep.name}).")

    def load_checkpoint(self):
        ck = torch.load(self._ck_path(), map_location=self.model.A.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.state = ck["train_state"]
        part = self.state.get("partial")
        if part is not None and part.get("best_state") is not None:
            dev = self.model.A.device
            part["best_state"] = {k: v.to(dev) for k, v in part["best_state"].items()}
        self._resume = part
        rng = ck.get("rng")
        if rng is not None:                          # checkpoints written before 15 Oct 2026 have no random state
            self.gen.set_state(rng["gen"])
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
        else:
            print("[bsce] note: old checkpoint without random state; the resumed run continues with a new minibatch "
                  "stream (not bit-identical to an uninterrupted run)")
        print(f"[bsce] resumed from {self.checkpoint}.pt ({len(self.state['done'])} stages done"
              + (f"; inside stage '{part['stage']}' at {part['where']})" if part else ")"))
