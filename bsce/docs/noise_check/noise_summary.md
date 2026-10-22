# Colored against white measurement noise (NMSE in dB; the same test channels and noise draw)

trained on = the noise model of the training data; data step 'matched' = NF-LADMM run with the exact data step of the test noise (same weights).

## SV_ULA

| method | trained on | data step | colored test 0 / 10 / 20 dB | mean | white test 0 / 10 / 20 dB | mean |
|---|---|---|---|---|---|---|
| amp | - | - | -3.48 / -10.94 / -15.06 | -10.27 | -5.20 / -11.27 / -14.91 | -10.80 |
| GM-LAMP-authors | colored | trained | -9.18 / -16.78 / -22.95 | -16.87 | -3.63 / -10.63 / -21.92 | -13.61 |
| LAMP-authors | colored | trained | -8.59 / -16.27 / -21.90 | -15.85 | -5.49 / -13.13 / -20.86 | -13.60 |
| NF-LADMM-SR-T4_paper_steps-4000 | colored | trained | -9.97 / -17.91 / -23.06 | -17.38 | -3.65 / -14.71 / -21.61 | -13.87 |
| NF-LADMM-SR-T4_paper_steps-4000 | white | trained | -9.02 / -17.01 / -21.63 | -16.36 | -9.78 / -17.53 / -21.82 | -16.87 |
| NF-LADMM-SR-T4_paper_steps-4000_noise-model-colored | white | trained | -8.10 / -17.18 / -22.10 | -16.34 | -7.34 / -15.77 / -21.23 | -15.19 |
| NF-LADMM-SR-T4_paper_steps-4000_white-step-adaptive | white | trained | -7.04 / -16.49 / -21.29 | -15.69 | -7.21 / -16.63 / -20.79 | -15.68 |
| NF-LADMM-SR-T4_paper_steps-4000 | white | matched | -9.92 / -17.93 / -22.87 | -17.35 | – | – |
| NF-LADMM-SR-T4_paper_steps-4000_white-step-adaptive | white | matched | -3.95 / -17.20 / -19.38 | -14.85 | – | – |
| NF-LADMM-SR-T4_paper_steps-4000 | colored | matched | – | – | -9.63 / -16.95 / -20.28 | -16.14 |

## SV_UPA

| method | trained on | data step | colored test 0 / 10 / 20 dB | mean | white test 0 / 10 / 20 dB | mean |
|---|---|---|---|---|---|---|
| amp | - | - | -2.68 / -8.22 / -10.24 | -7.50 | -3.83 / -8.17 / -10.14 | -7.70 |
| GM-LAMP-authors | colored | trained | -8.45 / -15.11 / -20.32 | -15.18 | -3.89 / -11.28 / -19.46 | -12.58 |
| LAMP-authors | colored | trained | -7.61 / -14.64 / -20.03 | -14.30 | -4.85 / -11.98 / -19.06 | -12.31 |
| NF-LADMM-SR-T4_paper_steps-4000 | colored | trained | -7.05 / -13.19 / -16.95 | -12.73 | -1.44 / -10.59 / -16.18 | -9.88 |
| NF-LADMM-SR-T4_paper_steps-4000 | white | trained | -6.11 / -12.02 / -14.84 | -11.40 | -6.53 / -12.19 / -14.89 | -11.61 |
| NF-LADMM-SR-T4_paper_steps-4000_noise-model-colored | white | trained | -5.26 / -12.63 / -16.08 | -11.83 | -5.03 / -11.58 / -15.64 | -11.10 |
| NF-LADMM-SR-T4_paper_steps-4000_white-step-adaptive | white | trained | -5.17 / -12.42 / -16.42 | -11.85 | -5.33 / -12.29 / -15.95 | -11.72 |
| NF-LADMM-SR-T4_paper_steps-4000 | white | matched | -6.63 / -12.79 / -16.25 | -12.27 | – | – |
| NF-LADMM-SR-T4_paper_steps-4000_white-step-adaptive | white | matched | -2.90 / -13.01 / -15.13 | -11.21 | – | – |
| NF-LADMM-SR-T4_paper_steps-4000 | colored | matched | – | – | -6.74 / -12.06 / -14.25 | -11.43 |
