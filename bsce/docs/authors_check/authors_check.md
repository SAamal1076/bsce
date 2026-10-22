# bsce against the authors' files (GM-LAMP data and paper protocol)

## Measurement files (y = A(x + n), the authors' A, SNR of every chunk estimated from y - A x)

| data set | train / val / test | file | chunks | SNR (dB) | max error (dB) | |
|---|---|---|---|---|---|---|
| SV_ULA | 80,000 / 2,000 / 2,000 | training 0to10dB | 10 | 9.1, 0.63, 9.28, 0.21, 6.38, 7.33, 1.5, 1.19, 5.85, 5.14 | 0.009 | OK |
| SV_ULA | 80,000 / 2,000 / 2,000 | training 10to20dB | 10 | 13.56, 11.87, 17.8, 11.31, 18.93, 11.83, 13.38, 16.94, 13.46, 12.08 | 0.007 | OK |
| SV_ULA | 80,000 / 2,000 / 2,000 | validation 0to10dB | 1 | 10 | 0.006 | OK |
| SV_ULA | 80,000 / 2,000 / 2,000 | validation 10to20dB | 1 | 10 | 0.010 | OK |
| SV_UPA | 80,000 / 2,000 / 2,000 | training 0to10dB | 10 | 9.1, 0.63, 9.28, 0.21, 6.38, 7.33, 1.5, 1.19, 5.85, 5.14 | 0.009 | OK |
| SV_UPA | 80,000 / 2,000 / 2,000 | training 10to20dB | 10 | 13.56, 11.87, 17.8, 11.31, 18.93, 11.83, 13.38, 16.94, 13.46, 12.08 | 0.007 | OK |
| SV_UPA | 80,000 / 2,000 / 2,000 | validation 0to10dB | 1 | 10 | 0.006 | OK |
| SV_UPA | 80,000 / 2,000 / 2,000 | validation 10to20dB | 1 | 10 | 0.010 | OK |
| DeepMIMO_ULA | 50,000 / 2,000 / 2,000 | training 0to10dB | 10 | 9.1, 0.63, 9.28, 0.21, 6.38, 7.33, 1.5, 1.19, 5.85, 5.14 | 0.013 | OK |
| DeepMIMO_ULA | 50,000 / 2,000 / 2,000 | training 10to20dB | 10 | 13.56, 11.87, 17.8, 11.31, 18.93, 11.83, 13.38, 16.94, 13.46, 12.08 | 0.016 | OK |
| DeepMIMO_ULA | 50,000 / 2,000 / 2,000 | validation 0to10dB | 1 | 10 | 0.016 | OK |
| DeepMIMO_ULA | 50,000 / 2,000 / 2,000 | validation 10to20dB | 1 | 10 | 0.003 | OK |
| DeepMIMO_UPA | 50,000 / 2,000 / 2,000 | training 0to10dB | 10 | 9.1, 0.63, 9.28, 0.21, 6.38, 7.33, 1.5, 1.19, 5.85, 5.14 | 0.013 | OK |
| DeepMIMO_UPA | 50,000 / 2,000 / 2,000 | training 10to20dB | 10 | 13.56, 11.87, 17.8, 11.31, 18.93, 11.83, 13.38, 16.94, 13.46, 12.08 | 0.016 | OK |
| DeepMIMO_UPA | 50,000 / 2,000 / 2,000 | validation 0to10dB | 1 | 10 | 0.016 | OK |
| DeepMIMO_UPA | 50,000 / 2,000 / 2,000 | validation 10to20dB | 1 | 10 | 0.003 | OK |

## The authors' published networks on our validation sets (10 dB)

Authors = the validation NMSE they logged at the end of training (their validation set).
sd = sampling spread of a 2000-channel NMSE (bootstrap); |z| < 2: as close as two random validation sets.

| network | authors (dB) | ours (dB) | difference (dB) | sd (dB) | z |
|---|---|---|---|---|---|
| LAMP SV_ULA 0to10dB | -14.655 | -14.635 | +0.020 | 0.050 | +0.28 |
| LAMP SV_ULA 10to20dB | -16.297 | -16.333 | -0.036 | 0.059 | -0.43 |
| GM-LAMP SV_ULA 0to10dB | -16.260 | -16.029 | +0.231 | 0.116 | +1.40 |
| GM-LAMP SV_ULA 10to20dB | -16.799 | -16.760 | +0.038 | 0.065 | +0.42 |
| LAMP SV_UPA 0to10dB | -13.023 | -13.114 | -0.091 | 0.049 | -1.30 |
| LAMP SV_UPA 10to20dB | -14.711 | -14.767 | -0.056 | 0.056 | -0.71 |
| GM-LAMP SV_UPA 0to10dB | -14.356 | -14.350 | +0.006 | 0.052 | +0.08 |
| GM-LAMP SV_UPA 10to20dB | -15.054 | -15.209 | -0.155 | 0.056 | -1.94 |
| LAMP DeepMIMO_ULA 0to10dB | -20.443 | -20.562 | -0.119 | 0.051 | -1.64 |
| LAMP DeepMIMO_ULA 10to20dB | -20.788 | -20.957 | -0.168 | 0.037 | -3.18 |
| GM-LAMP DeepMIMO_ULA 0to10dB | -21.343 | -21.585 | -0.242 | 0.078 | -2.18 |
| GM-LAMP DeepMIMO_ULA 10to20dB | -21.475 | -21.649 | -0.174 | 0.040 | -3.08 |
| LAMP DeepMIMO_UPA 0to10dB | -19.223 | -19.241 | -0.018 | 0.044 | -0.29 |
| LAMP DeepMIMO_UPA 10to20dB | -19.496 | -19.561 | -0.065 | 0.038 | -1.21 |
| GM-LAMP DeepMIMO_UPA 0to10dB | -20.862 | -20.891 | -0.029 | 0.050 | -0.41 |
| GM-LAMP DeepMIMO_UPA 10to20dB | -21.308 | -21.358 | -0.050 | 0.045 | -0.80 |

![validation](validation_vs_authors.png)
