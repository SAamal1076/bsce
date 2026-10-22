# 3GPP data check (3GPP_38.901_UMi, 28 GHz, 38.901 v19 tables)

Generator: bsce built-in (NumPy), 3GPP TR 38.901 v19.2.

## Standard conformance

LOS probability: generated 0.395, expected 0.393 (binomial sd 0.005).

| state | LSP | n | median drawn | median 38.901 | offset (dec) | log-sd drawn | log-sd 38.901 |
|---|---|---|---|---|---|---|---|
| LOS | DS (ns) | 3946 | 27.1 | 28.6 | -0.024 | 0.391 | 0.390 |
| LOS | ASD (deg) | 3946 | 13.1 | 13.7 | -0.020 | 0.420 | 0.407 |
| LOS | ASA (deg) | 3946 | 35.1 | 36.1 | -0.012 | 0.292 | 0.291 |
| LOS | ZSD (deg) | 3946 | 1.23 | 1.22 | +0.004 | 0.347 | 0.350 |
| LOS | ZSA (deg) | 3946 | 4.42 | 4.46 | -0.004 | 0.253 | 0.246 |
| NLOS | DS (ns) | 6054 | 67.9 | 64.3 | +0.024 | 0.506 | 0.498 |
| NLOS | ASD (deg) | 6054 | 15.2 | 15.5 | -0.008 | 0.472 | 0.476 |
| NLOS | ASA (deg) | 6054 | 45.8 | 45.5 | +0.003 | 0.341 | 0.343 |
| NLOS | ZSD (deg) | 6054 | 0.842 | 0.871 | -0.020 | 0.351 | 0.350 |
| NLOS | ZSA (deg) | 6054 | 7.56 | 7.52 | +0.002 | 0.276 | 0.277 |

K-factor (LOS): median 9.27 dB, sd 4.82 dB; 38.901: 9 dB, 5 dB.

## Beamspace (test sets)

| data set | beams for 95 % (median, p10-p90) | top-1 energy | top-16 energy |
|---|---|---|---|
| SV UPA | 30 (12-53) | 0.382 | 0.904 |
| DeepMIMO UPA | 15 (4-33) | 0.583 | 0.948 |
| 3GPP UPA | 17 (7-39) | 0.412 | 0.937 |

## LMMSE NMSE (dB)

| data set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| SV UPA | -3.19 | -5.18 | -7.41 | -9.57 | -11.38 |
| DeepMIMO UPA | -9.01 | -12.66 | -16.64 | -20.80 | -25.03 |
| 3GPP UPA | -5.41 | -8.08 | -11.04 | -14.12 | -17.09 |
