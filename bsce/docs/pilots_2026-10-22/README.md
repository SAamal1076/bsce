# Pilot experiments of 21–22 Oct 2026 (CPU, small data: indicative only)

Two scripts, run from anywhere with the package folder as first argument. Every run uses GM-LAMP's CS matrix
(`reference/gmlamp/CE_SV/CSmatrix256128.mat`), y = A (x + n), the layer-wise (borgerding) schedule, and is tested on
2000 test channels with the test noise of `evaluate.py` (seed 2026 + 1000 + i). Each script saves after every run and
resumes after an interruption.

| script | what it compares |
|---|---|
| `ladmm_ideas_small.py <bsce> <T> <out.json> [n_train] [maxit] [<data prefix>]` | LADMM, adaptive threshold (median), + residual noise level, + debias layer; 3 seeds |
| `ladmm_warm_small.py <bsce> <out.json> [n_train]` | LADMM and LADMM-AT, cold vs warm start (`train.warm_start`); T = 3, seed 0 |

## Results (`results/`, test NMSE in dB)

| file | setting |
|---|---|
| `T1.*` | SV UPA, T = 1, 500 training channels, 5000 iterations per stage |
| `T2.*` | SV UPA, T = 2, 500 training channels (all variants over-fit) |
| `T2_n1000_it2000.*` | SV UPA, T = 2, 1000 channels, 2000 iterations per stage |
| `T2_3gpp_n1000_it2000.json`, `T2_3gpp.log` | 3GPP UPA (built-in generator), the same |
| `warm_T3.*` | cold vs warm, T = 3, 1000 channels (over-fits after layer 2) |
| `warm_T3_n8000.*` | cold vs warm, T = 3, 8000 channels |

Main findings:
- The noise-adaptive threshold (LADMM-AT) beats LADMM in every run (T = 2, 1000 channels: +3.2 dB at 0 dB, +0.9 dB at
  20 dB; T = 3, 8000 channels: +1.6 dB at 0 dB, +1.2 dB at 20 dB).
- The residual noise level and the debias layer did not help with 1000 training channels (SV and 3GPP).
- Warm start: no difference with 1000 channels; with 8000 channels the cold start is better for both LADMM
  (0.15–0.29 dB) and LADMM-AT (0.07–0.14 dB), on validation too. A one-stage test of GM-LAMP (review round 2, D1)
  showed the opposite for GM-LAMP (warm about 3 dB better). Hence `run_suite.py --warm-start best`: each method is
  trained both ways and the variant with the better validation NMSE is kept.
- Every pilot stopped stages at the iteration cap: the networks are under-trained; the full-data, full-budget runs
  decide.
