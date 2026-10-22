"""scripts/check_vs_authors.py: its NumPy port of the authors' TensorFlow network (define_networks.build_LAMP +
define_shrinkage) gives the same output as bsce's LAMP / GM-LAMP (which are verified against the authors' code,
tests/results/summary.md), and its stage-log reader reads train_log.json. Synthetic data, no reference/ needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import check_vs_authors as C  # noqa: E402
from bsce.models import build_model  # noqa: E402


@pytest.mark.parametrize("model", ["LAMP", "GM-LAMP"])
def test_port_equals_bsce(model):
    r = np.random.default_rng(1)
    M, N, K = 32, 64, 50
    A = r.choice([-1.0, 1.0], (M, N)) / np.sqrt(M)
    torch.manual_seed(0)
    net = build_model(dict(model=model, T=3, precision="float64", K=4), A)
    with torch.no_grad():                         # move away from the init so every term matters
        for n, p in net.named_parameters():
            p.add_(0.05 * torch.randn_like(p) * (p.abs().mean() + 0.1))
            if model == "LAMP" and n.startswith("theta"):
                p.fill_(0.3)                      # a threshold that keeps part of the entries
    x = np.zeros((N, K), complex)
    for k in range(K):
        i = r.choice(N, 3, replace=False)
        x[i, k] = (r.standard_normal(3) + 1j * r.standard_normal(3)) * 3
    y = A @ (x + 0.1 * (r.standard_normal((N, K)) + 1j * r.standard_normal((N, K))))
    Bs = [net.P(f"B_{t}").detach().numpy() for t in range(3)]
    th = [net.P(f"theta_{t}").detach().numpy() for t in range(3)]
    ours = C.authors_net(y, A, Bs, th, model == "GM-LAMP")
    ref = net.forward_stages(torch.as_tensor(y))[-1].detach().numpy()
    assert np.count_nonzero(np.abs(ref) > 1e-6) > K          # not a trivial all-zero output
    assert np.max(np.abs(ours - ref)) / np.max(np.abs(ref)) < 1e-12


def test_stage_log_from_train_log(tmp_path):
    run = tmp_path / "LAMP-T1_paper"
    run.mkdir()
    log = [dict(stage="LAMP-soft linear T=1", nmse_db=-14.2, iters=9000)]
    (run / "train_log.json").write_text(json.dumps({"bands": {"0to10dB": {"stage_log": log}}}))
    assert C.stage_log_of(run, "0to10dB") == log
