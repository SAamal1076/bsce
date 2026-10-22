"""Regression tests for the fixes of the 21 Oct 2026 review (docs/REVIEW_2026-10-21.md)."""
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bsce.config import apply_overrides  # noqa: E402
from bsce.cost import macs_per_channel  # noqa: E402
from bsce.export import original_format_dict  # noqa: E402
from bsce.models import build_model  # noqa: E402
from bsce.models.graph_ladmm import estimate_edge_phases  # noqa: E402
from bsce.train.trainer import Trainer, TrainConfig  # noqa: E402


def _A(M=16, N=64, seed=0):
    return np.random.default_rng(seed).standard_normal((M, N)) / math.sqrt(M)


def _data(A, K, seed, snr_db=10.0):
    g = np.random.default_rng(seed)
    N = A.shape[1]
    x = np.zeros((N, K), complex)
    for k in range(K):
        x[g.choice(N, 3, replace=False), k] = g.standard_normal(3) + 1j * g.standard_normal(3)
    s = math.sqrt(10 ** (-snr_db / 10) / 2)
    y = A @ (x + s * (g.standard_normal(x.shape) + 1j * g.standard_normal(x.shape)))
    return x, y


def test_set_overrides_e_notation():
    c = apply_overrides({}, ["train.steps=2e4", "train.lr=8e-4", "train.refinements=[5e-1,1e-1]", "name=a.b", "k=3"])
    assert c["train"]["steps"] == 20000 and isinstance(c["train"]["steps"], int)
    assert c["train"]["lr"] == 8e-4 and c["train"]["refinements"] == [0.5, 0.1]
    assert c["name"] == "a.b" and c["k"] == 3


def test_orig_mat_only_for_scalar_threshold():
    A = _A()
    for thr in ("group", "group_k", "group_kc"):
        m = build_model(dict(model="LADMM", T=2, array="ULA", threshold=thr), A)
        assert original_format_dict(m) is None, thr
    assert original_format_dict(build_model(dict(model="LADMM", T=2, array="ULA"), A)) is not None


def test_theory_edge_phase_non_square_upa():
    N1, N2 = 4, 16
    m = build_model(dict(model="Graph-LADMM", T=2, array="UPA", N1=N1, N2=N2, edge_phase="theory"), _A(N=N1 * N2))
    want = {0: -math.pi / N1, 1: -math.pi / N2}
    for (ax, _), ph in zip(m.classes, m.phase.detach().cpu().numpy()):
        assert abs(ph - want[ax]) < 1e-6
    # the same size per axis as the phases measured on element-0 reference channels (the sign depends on the DFT
    # convention of this toy channel)
    g = np.random.default_rng(1)
    K = 400
    u, v = g.uniform(-1, 1, K), g.uniform(-1, 1, K)
    a, b = np.arange(N1), np.arange(N2)
    X = np.stack([np.kron(np.fft.fft(np.exp(1j * np.pi * a * u[k]), norm="ortho"),
                          np.fft.fft(np.exp(1j * np.pi * b * v[k]), norm="ortho")) for k in range(K)], 1)
    est = estimate_edge_phases(X, "UPA", N1, N2)
    for c, (ph, _, _) in est.items():
        if not c[1]:                                   # interior edges
            assert abs(abs(ph) - abs(want[c[0]])) < 0.05, (c, ph)


def test_macs_real_matrix_cast_to_complex():
    A = _A(128, 256)
    nf = build_model(dict(model="NF-LADMM", T=4, array="ULA"), A)
    lp = build_model(dict(model="LAMP", T=4, shrink="soft", precision="float32"), A)
    assert macs_per_channel(lp)[0] == 2 * 256 * 128 + 3 * 4 * 256 * 128          # unchanged
    # NF-LADMM multiplies real A, A^+ with complex vectors: 2 real MACs per entry, not 4
    with torch.no_grad():
        Ac = nf.A.to(nf.cdtype)
    from bsce.cost import _complex_factor
    assert _complex_factor(Ac, Ac) == 1.0 and _complex_factor(Ac, Ac * 1j + 1) == 2.0
    assert macs_per_channel(nf)[0] < 4.0e6


def _admm(A):
    torch.manual_seed(0)
    return build_model(dict(model="ADMM-Net", T=2, precision="float64"), A)


def test_end2end_honours_grad_clip():
    A = _A()
    xt, yt = _data(A, 256, 1)
    xv, yv = _data(A, 64, 2)
    out = []
    for clip in (None, 1e-9):
        m = _admm(A)
        w0 = {k: v.clone() for k, v in m.state_dict().items()}
        cfg = TrainConfig(schedule="end2end", epochs=1, optimizer="adam_torch", lr=1e-2, batch_size=64, loss="nmse",
                          val_round=None, seed=1, verbose=False, grad_clip=clip)
        Trainer(m, cfg, xt, yt, xv, yv).run()
        out.append(max(float((m.state_dict()[k] - w0[k]).abs().max()) for k in w0 if w0[k].is_floating_point()
                       or w0[k].is_complex()))
    assert out[1] < 0.1 * out[0], out


def test_end2end_lr_list_keeps_the_better_weights():
    A = _A()
    xt, yt = _data(A, 256, 1)
    xv, yv = _data(A, 64, 2)
    m = _admm(A)
    cfg = TrainConfig(schedule="end2end", epochs=6, lr_list=[0.02, 5.0], optimizer="adam_torch", batch_size=64,
                      loss="nmse", val_round=None, seed=1, verbose=False, restore_best=True, patience=1)
    tr = Trainer(m, cfg, xt, yt, xv, yv)
    st = tr.run()
    first = [e["nmse_db"] for e in st["log"] if e["stage"].startswith("end2end lr=0.02 epoch")]
    final = 10 * math.log10(tr.validate(m.stage_plan()[-1].index))
    assert first and final <= min(first) + 1e-9, (final, min(first))


def test_aggregate_seeds_groups_polish_columns(tmp_path):
    f = tmp_path / "nmse_vs_snr.csv"
    cols = ["SNR_dB", "L-T6", "L-T6_s1", "L-T6+polish", "L-T6_s1+polish"]
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow([0, -1, -2, -3, -4])
        w.writerow([10, -5, -6, -7, -8])
    r = subprocess.run([sys.executable, "scripts/aggregate_seeds.py", "--csv", str(f)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "n=2" in r.stdout and "L-T6+polish" in r.stdout and "L-T6_s1+polish" not in r.stdout, r.stdout
