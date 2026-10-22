"""Reference run of the ORIGINAL THADMM-Net repo ADMM-Net (PyTorch) - forward, grads, Adam."""
import sys
from pathlib import Path

import numpy as np
import torch

REF = Path(__file__).resolve().parents[2] / "reference" / "thadmmnet"
sys.path.insert(0, str(REF))
from models.admmnet import AdmmNet          # original, unchanged
from utils.utils import nmse                # original loss

inp = dict(np.load(sys.argv[1], allow_pickle=True))
A = torch.tensor(inp["A"].astype(np.complex128))
T = int(inp["T"])
B = A.T.conj() @ A
net = AdmmNet(B, A, num_layers=T, beta=float(inp["beta"]), rho=float(inp["rho"]), device="cpu")
with torch.no_grad():
    for t, cell in enumerate(net.layers):
        cell.W.copy_(torch.tensor(inp[f"W_{t}"]))
        cell.rho.copy_(torch.tensor(inp[f"rho_{t}"]))
        cell.S.beta.copy_(torch.tensor(inp[f"beta_{t}"]))
y = torch.tensor(inp["y"].astype(np.complex128))
x = torch.tensor(inp["x"].astype(np.complex128))
out = {"x_est": net(y).detach().numpy()}
loss = nmse(net(y), x)
net.zero_grad()
loss.backward()
for t, cell in enumerate(net.layers):
    for k, p in (("W", cell.W), ("rho", cell.rho), ("beta", cell.S.beta)):
        if p.grad is not None:          # last-layer beta has no gradient (output is x)
            out[f"grad_{k}_{t}"] = p.grad.numpy()
opt = torch.optim.Adam(net.parameters(), float(inp["lr"]))
for _ in range(int(inp["nsteps"])):
    loss = nmse(net(y), x)
    net.zero_grad()
    loss.backward()
    opt.step()
for t, cell in enumerate(net.layers):
    out[f"after_W_{t}"] = cell.W.detach().numpy()
    out[f"after_rho_{t}"] = cell.rho.detach().numpy()
    out[f"after_beta_{t}"] = cell.S.beta.detach().numpy()
out["beta_dtype"] = str(net.layers[0].S.beta.dtype)
np.savez(sys.argv[2], **out)
