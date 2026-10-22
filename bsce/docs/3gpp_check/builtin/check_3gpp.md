# 3GPP data check (3GPP_38.901_UMi, 28 GHz, 38.901 v16 tables)

Generator: bsce built-in (NumPy), 3GPP TR 38.901 v16.1.

## Standard conformance

LOS probability: generated 0.395, expected 0.393 (binomial sd 0.005).

| state | LSP | n | median drawn | median 38.901 | offset (dec) | log-sd drawn | log-sd 38.901 |
|---|---|---|---|---|---|---|---|
| LOS | DS (ns) | 3946 | 30.6 | 32.3 | -0.024 | 0.381 | 0.380 |
| LOS | ASD (deg) | 3946 | 13.1 | 13.7 | -0.020 | 0.423 | 0.410 |
| LOS | ASA (deg) | 3946 | 39.8 | 41 | -0.013 | 0.302 | 0.300 |
| LOS | ZSD (deg) | 3946 | 1.23 | 1.22 | +0.004 | 0.347 | 0.350 |
| LOS | ZSA (deg) | 3946 | 3.8 | 3.83 | -0.004 | 0.289 | 0.282 |
| NLOS | DS (ns) | 6054 | 69.7 | 65.9 | +0.024 | 0.523 | 0.514 |
| NLOS | ASD (deg) | 6054 | 15.3 | 15.6 | -0.008 | 0.486 | 0.491 |
| NLOS | ASA (deg) | 6054 | 49.7 | 49.3 | +0.003 | 0.370 | 0.373 |
| NLOS | ZSD (deg) | 6054 | 0.842 | 0.871 | -0.020 | 0.351 | 0.350 |
| NLOS | ZSA (deg) | 6054 | 7.31 | 7.27 | +0.002 | 0.306 | 0.308 |

K-factor (LOS): median 9.27 dB, sd 4.82 dB; 38.901: 9 dB, 5 dB.

## Beamspace (test sets)

| data set | beams for 95 % (median, p10-p90) | top-1 energy | top-16 energy |
|---|---|---|---|
| SV UPA | 30 (12-53) | 0.382 | 0.904 |
| DeepMIMO UPA | 15 (4-33) | 0.583 | 0.948 |
| 3GPP UPA | 17 (7-39) | 0.413 | 0.937 |
| SV ULA | 9 (4-16) | 0.481 | 0.973 |
| DeepMIMO ULA (z axis) | 7 (2-11) | 0.627 | 0.985 |
| 3GPP ULA (y axis) | 65 (7-118) | 0.301 | 0.655 |
| 3GPP ULA (z axis) | 8 (4-16) | 0.513 | 0.977 |

## LMMSE NMSE (dB)

| data set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| SV UPA | -3.19 | -5.18 | -7.41 | -9.57 | -11.38 |
| DeepMIMO UPA | -9.01 | -12.66 | -16.64 | -20.80 | -25.03 |
| 3GPP UPA | -5.41 | -8.08 | -11.05 | -14.12 | -17.10 |
| SV ULA | -1.55 | -2.44 | -3.06 | -3.34 | -3.44 |
| DeepMIMO ULA (z axis) | -7.58 | -11.13 | -15.03 | -19.14 | -23.45 |
| 3GPP ULA (y axis) | -1.46 | -2.45 | -3.18 | -3.54 | -3.67 |
| 3GPP ULA (z axis) | -5.74 | -8.41 | -11.31 | -14.31 | -17.20 |

## Transfer of the authors' GM-LAMP (NMSE, dB)

| network, test set, SNR | NMSE |
|---|---|
| GM-LAMP(SV UPA, 0to10dB) on SV UPA at 5 dB | -12.66 |
| GM-LAMP(SV UPA, 0to10dB) on DeepMIMO UPA at 5 dB | -7.80 |
| GM-LAMP(SV UPA, 0to10dB) on 3GPP UPA at 5 dB | -2.96 |
| GM-LAMP(SV UPA, 10to20dB) on SV UPA at 15 dB | -18.60 |
| GM-LAMP(SV UPA, 10to20dB) on DeepMIMO UPA at 15 dB | -9.88 |
| GM-LAMP(SV UPA, 10to20dB) on 3GPP UPA at 15 dB | -0.70 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on SV UPA at 5 dB | 0.30 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on DeepMIMO UPA at 5 dB | -18.05 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on 3GPP UPA at 5 dB | 1.59 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on SV UPA at 15 dB | -0.42 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on DeepMIMO UPA at 15 dB | -25.84 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on 3GPP UPA at 15 dB | 0.61 |
| GM-LAMP(SV ULA, 0to10dB) on SV ULA at 5 dB | -13.74 |
| GM-LAMP(SV ULA, 0to10dB) on DeepMIMO ULA (z axis) at 5 dB | -9.69 |
| GM-LAMP(SV ULA, 0to10dB) on 3GPP ULA (y axis) at 5 dB | -2.55 |
| GM-LAMP(SV ULA, 0to10dB) on 3GPP ULA (z axis) at 5 dB | -8.75 |
| GM-LAMP(SV ULA, 10to20dB) on SV ULA at 15 dB | -20.79 |
| GM-LAMP(SV ULA, 10to20dB) on DeepMIMO ULA (z axis) at 15 dB | -12.93 |
| GM-LAMP(SV ULA, 10to20dB) on 3GPP ULA (y axis) at 15 dB | -1.08 |
| GM-LAMP(SV ULA, 10to20dB) on 3GPP ULA (z axis) at 15 dB | -12.59 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on SV ULA at 5 dB | -0.44 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on DeepMIMO ULA (z axis) at 5 dB | -18.96 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on 3GPP ULA (y axis) at 5 dB | -0.40 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on 3GPP ULA (z axis) at 5 dB | -6.11 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on SV ULA at 15 dB | 0.34 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on DeepMIMO ULA (z axis) at 15 dB | -26.55 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on 3GPP ULA (y axis) at 15 dB | 0.01 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on 3GPP ULA (z axis) at 15 dB | -8.65 |
