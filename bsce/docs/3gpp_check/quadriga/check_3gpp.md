# 3GPP data check (3GPP_38.901_UMi, 28 GHz, 38.901 v16 tables)

Generator: QuaDRiGa v2.8.1 (3GPP TR 38.901 v16 tables).

## Standard conformance

LOS probability: generated 0.390, expected 0.394 (binomial sd 0.005).

| state | LSP | n | median drawn | median 38.901 | offset (dec) | log-sd drawn | log-sd 38.901 | median of paths |
|---|---|---|---|---|---|---|---|---|
| LOS | DS (ns) | 3897 | 32.5 | 32.3 | +0.002 | 0.385 | 0.380 | 15.3 |
| LOS | ASD (deg) | 3897 | 13.7 | 13.7 | -0.001 | 0.400 | 0.410 | 13 |
| LOS | ASA (deg) | 3897 | 40.6 | 41 | -0.004 | 0.295 | 0.300 | 27.8 |
| LOS | ZSD (deg) | 3897 | 1.26 | 1.25 | +0.010 | 0.352 | 0.350 | 1.84 |
| LOS | ZSA (deg) | 3897 | 3.86 | 3.83 | +0.002 | 0.276 | 0.282 | 5.65 |
| NLOS | DS (ns) | 6103 | 64.7 | 65.9 | -0.008 | 0.497 | 0.514 | 61.4 |
| NLOS | ASD (deg) | 6103 | 16.1 | 15.6 | +0.013 | 0.487 | 0.491 | 15.9 |
| NLOS | ASA (deg) | 6103 | 48.8 | 49.3 | -0.004 | 0.369 | 0.373 | 44.8 |
| NLOS | ZSD (deg) | 6103 | 0.867 | 0.872 | -0.004 | 0.345 | 0.350 | 1.1 |
| NLOS | ZSA (deg) | 6103 | 7.33 | 7.27 | +0.003 | 0.295 | 0.308 | 10 |

K-factor (LOS): median 8.96 dB, sd 5.09 dB; 38.901: 9 dB, 5 dB.

## Beamspace (test sets)

| data set | beams for 95 % (median, p10-p90) | top-1 energy | top-16 energy |
|---|---|---|---|
| SV UPA | 30 (12-53) | 0.382 | 0.904 |
| DeepMIMO UPA | 15 (4-33) | 0.583 | 0.948 |
| 3GPP UPA | 24 (10-49) | 0.376 | 0.907 |
| SV ULA | 9 (4-16) | 0.481 | 0.973 |
| DeepMIMO ULA (z axis) | 7 (2-11) | 0.627 | 0.985 |
| 3GPP ULA (y axis) | 98 (8-143) | 0.293 | 0.622 |
| 3GPP ULA (z axis) | 9 (4-17) | 0.472 | 0.974 |

## LMMSE NMSE (dB)

| data set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| SV UPA | -3.19 | -5.18 | -7.41 | -9.57 | -11.38 |
| DeepMIMO UPA | -9.01 | -12.66 | -16.64 | -20.80 | -25.03 |
| 3GPP UPA | -5.03 | -7.62 | -10.50 | -13.44 | -16.19 |
| SV ULA | -1.55 | -2.44 | -3.06 | -3.34 | -3.44 |
| DeepMIMO ULA (z axis) | -7.58 | -11.13 | -15.03 | -19.14 | -23.45 |
| 3GPP ULA (y axis) | -1.41 | -2.35 | -3.02 | -3.34 | -3.45 |
| 3GPP ULA (z axis) | -5.30 | -7.94 | -10.77 | -13.69 | -16.44 |

## Transfer of the authors' GM-LAMP (NMSE, dB)

| network, test set, SNR | NMSE |
|---|---|
| GM-LAMP(SV UPA, 0to10dB) on SV UPA at 5 dB | -12.66 |
| GM-LAMP(SV UPA, 0to10dB) on DeepMIMO UPA at 5 dB | -7.80 |
| GM-LAMP(SV UPA, 0to10dB) on 3GPP UPA at 5 dB | -2.71 |
| GM-LAMP(SV UPA, 10to20dB) on SV UPA at 15 dB | -18.60 |
| GM-LAMP(SV UPA, 10to20dB) on DeepMIMO UPA at 15 dB | -9.88 |
| GM-LAMP(SV UPA, 10to20dB) on 3GPP UPA at 15 dB | -0.69 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on SV UPA at 5 dB | 0.30 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on DeepMIMO UPA at 5 dB | -18.05 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on 3GPP UPA at 5 dB | 1.60 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on SV UPA at 15 dB | -0.42 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on DeepMIMO UPA at 15 dB | -25.84 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on 3GPP UPA at 15 dB | 0.61 |
| GM-LAMP(SV ULA, 0to10dB) on SV ULA at 5 dB | -13.74 |
| GM-LAMP(SV ULA, 0to10dB) on DeepMIMO ULA (z axis) at 5 dB | -9.69 |
| GM-LAMP(SV ULA, 0to10dB) on 3GPP ULA (y axis) at 5 dB | -2.36 |
| GM-LAMP(SV ULA, 0to10dB) on 3GPP ULA (z axis) at 5 dB | -8.28 |
| GM-LAMP(SV ULA, 10to20dB) on SV ULA at 15 dB | -20.79 |
| GM-LAMP(SV ULA, 10to20dB) on DeepMIMO ULA (z axis) at 15 dB | -12.93 |
| GM-LAMP(SV ULA, 10to20dB) on 3GPP ULA (y axis) at 15 dB | -0.69 |
| GM-LAMP(SV ULA, 10to20dB) on 3GPP ULA (z axis) at 15 dB | -12.17 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on SV ULA at 5 dB | -0.44 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on DeepMIMO ULA (z axis) at 5 dB | -18.96 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on 3GPP ULA (y axis) at 5 dB | -0.39 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on 3GPP ULA (z axis) at 5 dB | -4.95 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on SV ULA at 15 dB | 0.34 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on DeepMIMO ULA (z axis) at 15 dB | -26.55 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on 3GPP ULA (y axis) at 15 dB | 0.06 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on 3GPP ULA (z axis) at 15 dB | -6.50 |
