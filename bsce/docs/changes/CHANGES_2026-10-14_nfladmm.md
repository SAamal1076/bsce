# 14 Oct 2026 (2): NF-LADMM as a compared method, any SNR range, depth selection for LADMM and NF-LADMM

## NF-LADMM (NF-LADMM_proposal.md)
**`bsce/models/nfladmm.py`, model `NF-LADMM`.**
- Exact noise-folded x-step `x = v + beta A⁺(y − Av)`.
- Learned prior: monotone radial shrinkage, plus a shared circular CNN for `denoiser: conv`.
  - 1-D kernel 9 for the ULA.
  - 2-D 5×5 on the 16×16 beam grid for the UPA.
- σ_t from the residual plus the known noise variance (`sigma: known`), or from the residual alone (`sigma: residual`).
- Parameters at T = 8: 3,434 (ULA) / 9,066 (UPA) for -S, and 160 for -C.

**Checks (`tests/test_nfladmm.py`, 7 passed).**
- Same output as the proposal's module (`reference/proposals/nf_ladmm.py`) with the same weights: relative difference 4e-7 to 1e-6, for ULA/UPA × radial/conv.
- The closed-form x-step equals the explicit solve to 1e-12.
- Every parameter receives a gradient.
- Save/load round trip.
- Online-noise training lowers the validation NMSE.

**Configs.**
- `configs/train/nfladmm_s.yaml` and `nfladmm_c.yaml`. Schedule `steps`:
  - end-to-end Adam, cosine decay 1e-3 → 5e-5, gradient clip 1
  - batch 128, loss log-NSE, 30,000 steps
  - online noise, best validation state kept
- `evaluate.py` passes the test noise variance to models that need it.

## Any SNR range
- **`train.py --snr <edges>`** sets any SNR bands:
  - `0 20` = one network
  - `0 10 20` = two networks
  - `-5 25`, …
  
  Missing measurement files are made automatically: 100 SNR chunks ~ U(band), plus `snr_db` per column.
- **`train.py --noise online`**: fresh noise every minibatch, one SNR per sample ~ U(band), and a fixed generated validation set (the same for every method with that band). Works for every model, LADMM included.
- **`run_all.py`** passes `--snr` / `--noise` to every config. **`make_sv_measurements.py --snr`** makes the files ahead of time.
- **`evaluate.py`** finds every `<tag>_<band>.pt` of a run folder. The range comes from the scenario or from the band name (`0to20dB`, `m5to0dB`).
  - One-network runs and band-pair runs share one table.
  - If a folder has both, the narrowest band containing the SNR is used.
  - An SNR outside every band uses the nearest band.
- **Trainer options**: `noise`, `snr_range`, `schedule: steps`, `steps`, `lr_final_frac`, `grad_clip`, `val_every`, and `s2_train` / `s2_val`.
- **Loss** `log_nse`.

## Depth of LADMM and NF-LADMM: `scripts/select_depth.py`
- **Layer-wise configs (LADMM)**: one run at max T. The logged validation NMSE per depth gives the curve.
- **End-to-end configs (NF-LADMM)**: one run per T.
- **Output**: one plot per band with all methods, values printed and T* marked, plus csv/json.
- **Runbook**: S2 now uses it for both methods; S3 adds NF-LADMM-S/C per scenario; Phase D adds D3b (NF-LADMM on your `data/SV`). `status.py` follows.
