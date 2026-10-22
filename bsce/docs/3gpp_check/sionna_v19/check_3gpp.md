# 3GPP data check (3GPP_38.901_UMi, 28 GHz, 38.901 v19 tables)

Generator: Sionna 2.2.0 (sionna.phy.channel.tr38901.UMi, 3GPP TR 38.901 v19.2).

## Standard conformance

LOS probability: generated 0.398, expected 0.390 (binomial sd 0.007).

| state | LSP | n | median drawn | median 38.901 | offset (dec) | log-sd drawn | log-sd 38.901 |
|---|---|---|---|---|---|---|---|
| LOS | DS (ns) | 1990 | 29.3 | 28.6 | +0.010 | 0.379 | 0.390 |
| LOS | ASD (deg) | 1990 | 14.5 | 13.7 | +0.025 | 0.421 | 0.407 |
| LOS | ASA (deg) | 1990 | 36.6 | 36.1 | +0.006 | 0.282 | 0.291 |
| LOS | ZSD (deg) | 1990 | 1.14 | 1.2 | -0.018 | 0.351 | 0.350 |
| LOS | ZSA (deg) | 1990 | 4.51 | 4.46 | +0.005 | 0.240 | 0.246 |
| NLOS | DS (ns) | 3010 | 63.6 | 64.3 | -0.005 | 0.489 | 0.498 |
| NLOS | ASD (deg) | 3010 | 16 | 15.5 | +0.016 | 0.470 | 0.476 |
| NLOS | ASA (deg) | 3010 | 46.6 | 45.5 | +0.010 | 0.335 | 0.343 |
| NLOS | ZSD (deg) | 3010 | 0.913 | 0.871 | +0.019 | 0.344 | 0.350 |
| NLOS | ZSA (deg) | 3010 | 7.61 | 7.52 | +0.005 | 0.287 | 0.277 |

K-factor (LOS): median 8.91 dB, sd 4.93 dB; 38.901: 9 dB, 5 dB.

## Beamspace (test sets)

| data set | beams for 95 % (median, p10-p90) | top-1 energy | top-16 energy |
|---|---|---|---|
| SV UPA | 30 (12-53) | 0.382 | 0.904 |
| DeepMIMO UPA | 15 (4-33) | 0.583 | 0.948 |
| 3GPP UPA | 17 (8-40) | 0.410 | 0.935 |

## LMMSE NMSE (dB)

| data set | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| SV UPA | -3.16 | -5.13 | -7.35 | -9.50 | -11.30 |
| DeepMIMO UPA | -9.00 | -12.65 | -16.64 | -20.79 | -25.02 |
| 3GPP UPA | -5.51 | -8.20 | -11.16 | -14.23 | -17.28 |
