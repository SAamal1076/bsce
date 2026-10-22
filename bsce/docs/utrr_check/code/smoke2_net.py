"""Smoke test 2: the UTRR network (utrr_ref.UTRRNet) against an independent NumPy loop of the paper's equations.
  a  forward at initialisation (A_t = [Phi, -Phi], rho_t = 1, alpha_t = 0.1, no BN) = the unrolled iteration of eq. 23
     (K = 0 in layers 0..L-2, K in the last layer, beta = 1/2), float64
  b  number of trainable parameters = L (2MN + 2) without BN (the paper's count)
  c  every parameter receives a gradient
  d  forward(y, upto=t) = a separate (t+1)-layer network with the same first t+1 layers (truncation)
  e  complex wrapper = real and imaginary parts as two real samples
  f  forward time on this CPU
usage: python smoke2_net.py <bsce folder> <out folder>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utrr_ref import UTRRNet, topk_keep  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
Phi = sio.loadmat(root / "reference/gmlamp/CE_SV/CSmatrix256128.mat")["A"].astype(float)
X = sio.loadmat(root / "data/SV_ULA/SV_ULA_test_channel.mat")["x"][:, :64]
M, N = Phi.shape
rng = np.random.default_rng(0)
yc = Phi @ X + 0.05 * (rng.standard_normal((M, 64)) + 1j * rng.standard_normal((M, 64)))
yr = np.concatenate([yc.real, yc.imag], 1)
res = {}

# a: init forward vs NumPy
L, K = 30, 16
A = np.hstack([Phi, -Phi])
x0 = Phi.T @ yr
z = np.vstack([np.maximum(x0, 0), np.maximum(-x0, 0)])
for t in range(L):
    k = K if t == L - 1 else 0
    g = A.T @ (A @ z - yr) + 1.0 * (z - topk_keep(z, k))
    w = np.maximum(z - 0.1 * g, 0)
    z = (w + z) / 2
x_np = z[:N] - z[N:]
net = UTRRNet(Phi, L=L, K=K, rcc=True, bn=False, dtype=torch.float64)
with torch.no_grad():
    x_t = net(torch.tensor(yr)).numpy()
res["a_init_forward_max_abs_diff"] = float(np.abs(x_t - x_np).max())
res["a_init_nmse_dB"] = float(10 * np.log10(np.sum(np.abs((x_t[:, :64] + 1j * x_t[:, 64:]) - X) ** 2) / np.sum(np.abs(X) ** 2)))

# b: parameter count
res["b_params_no_bn"] = net.n_params()
res["b_params_paper_L(2MN+2)"] = L * (2 * M * N + 2)
res["b_params_with_bn"] = UTRRNet(Phi, L=L, K=K, bn=True).n_params()

# c: gradients
netb = UTRRNet(Phi, L=6, K=K, bn=True, dtype=torch.float64)
loss = torch.sum((netb(torch.tensor(yr)) - torch.tensor(np.concatenate([X.real, X.imag], 1))) ** 2)
loss.backward()
res["c_params_without_gradient"] = [n for n, p in netb.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]

# d: truncation
torch.manual_seed(1)
big = UTRRNet(Phi, L=8, K=K, bn=False, dtype=torch.float64)
with torch.no_grad():
    for p in big.parameters():
        p.add_(0.01 * torch.randn_like(p))
small = UTRRNet(Phi, L=5, K=K, bn=False, dtype=torch.float64)
sd = {k: v for k, v in big.state_dict().items() if k in small.state_dict()}
small.load_state_dict(sd)
with torch.no_grad():
    res["d_truncation_max_abs_diff"] = float((big(torch.tensor(yr), upto=4) - small(torch.tensor(yr))).abs().max())

# e: complex wrapper
with torch.no_grad():
    xc = net.forward_complex(torch.tensor(yc)).numpy()
res["e_complex_wrapper_max_abs_diff"] = float(np.abs(xc - (x_t[:, :64] + 1j * x_t[:, 64:])).max())

# f: speed
net32 = UTRRNet(Phi, L=30, K=K, bn=True)
yb = torch.randn(M, 128)
with torch.no_grad():
    net32.eval()
    t0 = time.time()
    for _ in range(5):
        net32(yb)
    res["f_forward_ms_per_128_real_samples_cpu"] = round((time.time() - t0) / 5 * 1000, 1)
net32.train()
t0 = time.time()
for _ in range(3):
    l = net32(yb).pow(2).sum()
    l.backward()
res["f_train_step_ms_per_128_real_samples_cpu"] = round((time.time() - t0) / 3 * 1000, 1)
res["torch_threads"] = torch.get_num_threads()
(out / "smoke2_net.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
