# 3GPP data check (3GPP_38.901_UMi, 28 GHz, 38.901 v16 tables)

Generator: Sionna 2.2.0 (sionna.phy.channel.tr38901.UMi, 3GPP TR 38.901 v16.1).

## Standard conformance

LOS probability: generated 0.395, expected 0.393 (binomial sd 0.005).

| state | LSP | n | median drawn | median 38.901 | offset (dec) | log-sd drawn | log-sd 38.901 |
|---|---|---|---|---|---|---|---|
| LOS | DS (ns) | 3950 | 33.2 | 32.3 | +0.012 | 0.386 | 0.380 |
| LOS | ASD (deg) | 3950 | 14 | 13.7 | +0.010 | 0.426 | 0.410 |
| LOS | ASA (deg) | 3950 | 42.4 | 41 | +0.015 | 0.295 | 0.300 |
| LOS | ZSD (deg) | 3950 | 1.19 | 1.23 | -0.017 | 0.346 | 0.350 |
| LOS | ZSA (deg) | 3950 | 3.78 | 3.83 | -0.007 | 0.283 | 0.282 |
| NLOS | DS (ns) | 6050 | 64.9 | 65.9 | -0.007 | 0.512 | 0.514 |
| NLOS | ASD (deg) | 6050 | 16 | 15.6 | +0.010 | 0.491 | 0.491 |
| NLOS | ASA (deg) | 6050 | 50.8 | 49.3 | +0.013 | 0.368 | 0.373 |
| NLOS | ZSD (deg) | 6050 | 0.898 | 0.871 | +0.012 | 0.352 | 0.350 |
| NLOS | ZSA (deg) | 6050 | 7.48 | 7.27 | +0.012 | 0.309 | 0.308 |

K-factor (LOS): median 8.95 dB, sd 5.23 dB; 38.901: 9 dB, 5 dB.

## Beamspace (test sets)

| data set | beams for 95 % (median, p10-p90) | top-1 energy | top-16 energy |
|---|---|---|---|
| SV UPA | 30 (12-53) | 0.382 | 0.904 |
| DeepMIMO UPA | 15 (4-33) | 0.583 | 0.948 |
| 3GPP UPA | 17 (7-38) | 0.411 | 0.937 |
| SV ULA | 9 (4-16) | 0.481 | 0.973 |
| DeepMIMO ULA (z axis) | 7 (2-11) | 0.627 | 0.985 |
| 3GPP ULA (y axis) | 62 (8-118) | 0.296 | 0.657 |
| 3GPP ULA (z axis) | 8 (4-17) | 0.512 | 0.976 |

## LMMSE NMSE (dB)

| data set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| SV UPA | -3.19 | -5.18 | -7.41 | -9.57 | -11.38 |
| DeepMIMO UPA | -9.01 | -12.66 | -16.64 | -20.80 | -25.03 |
| 3GPP UPA | -5.36 | -8.04 | -11.04 | -14.12 | -17.13 |
| SV ULA | -1.55 | -2.44 | -3.06 | -3.34 | -3.44 |
| DeepMIMO ULA (z axis) | -7.58 | -11.13 | -15.03 | -19.14 | -23.45 |
| 3GPP ULA (y axis) | -1.46 | -2.46 | -3.19 | -3.54 | -3.67 |
| 3GPP ULA (z axis) | -5.71 | -8.37 | -11.31 | -14.36 | -17.30 |

## Transfer of the authors' GM-LAMP (NMSE, dB)

| network, test set, SNR | NMSE |
|---|---|
| GM-LAMP(SV UPA, 0to10dB) on SV UPA at 5 dB | -12.66 |
| GM-LAMP(SV UPA, 0to10dB) on DeepMIMO UPA at 5 dB | -7.80 |
| GM-LAMP(SV UPA, 0to10dB) on 3GPP UPA at 5 dB | -3.01 |
| GM-LAMP(SV UPA, 10to20dB) on SV UPA at 15 dB | -18.60 |
| GM-LAMP(SV UPA, 10to20dB) on DeepMIMO UPA at 15 dB | -9.88 |
| GM-LAMP(SV UPA, 10to20dB) on 3GPP UPA at 15 dB | -0.69 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on SV UPA at 5 dB | 0.30 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on DeepMIMO UPA at 5 dB | -18.05 |
| GM-LAMP(DeepMIMO UPA, 0to10dB) on 3GPP UPA at 5 dB | 1.58 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on SV UPA at 15 dB | -0.42 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on DeepMIMO UPA at 15 dB | -25.84 |
| GM-LAMP(DeepMIMO UPA, 10to20dB) on 3GPP UPA at 15 dB | 0.61 |
| GM-LAMP(SV ULA, 0to10dB) on SV ULA at 5 dB | -13.74 |
| GM-LAMP(SV ULA, 0to10dB) on DeepMIMO ULA (z axis) at 5 dB | -9.69 |
| GM-LAMP(SV ULA, 0to10dB) on 3GPP ULA (y axis) at 5 dB | -2.54 |
| GM-LAMP(SV ULA, 0to10dB) on 3GPP ULA (z axis) at 5 dB | -8.72 |
| GM-LAMP(SV ULA, 10to20dB) on SV ULA at 15 dB | -20.79 |
| GM-LAMP(SV ULA, 10to20dB) on DeepMIMO ULA (z axis) at 15 dB | -12.93 |
| GM-LAMP(SV ULA, 10to20dB) on 3GPP ULA (y axis) at 15 dB | -1.10 |
| GM-LAMP(SV ULA, 10to20dB) on 3GPP ULA (z axis) at 15 dB | -12.38 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on SV ULA at 5 dB | -0.44 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on DeepMIMO ULA (z axis) at 5 dB | -18.96 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on 3GPP ULA (y axis) at 5 dB | -0.41 |
| GM-LAMP(DeepMIMO ULA, 0to10dB) on 3GPP ULA (z axis) at 5 dB | -6.14 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on SV ULA at 15 dB | 0.34 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on DeepMIMO ULA (z axis) at 15 dB | -26.55 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on 3GPP ULA (y axis) at 15 dB | 0.03 |
| GM-LAMP(DeepMIMO ULA, 10to20dB) on 3GPP ULA (z axis) at 15 dB | -8.85 |
