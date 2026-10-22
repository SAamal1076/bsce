# STATUS (scripts/status.py)

| step | task | done | missing files |
|---|---|---|---|
| S0a | data ready (4 scenarios, measurements) | DONE (4/4) |  |
| S0b | speed test: device and precision | todo (0/1) | outputs/speed_test.json |
| S1 | sanity: LAMP + GM-LAMP retrained on SV_ULA | todo (0/2) | outputs/SV_ULA/LAMP_gmlamp/train_log.json, outputs/SV_ULA/GM-LAMP-spread_gmlamp/train_log.json |
| S2 | depth of LADMM and NF-LADMM on SV_ULA, SV_UPA (select_depth.py) | todo (0/2) | outputs/depth_select/SV_ULA/depth_select.json, outputs/depth_select/SV_UPA/depth_select.json |
| S3-SV_ULA | main comparison SV_ULA (9 learned + NF-LADMM-S/C + AMP/OMP) + evaluation | todo (0/12) | outputs/SV_ULA/LAMP_gmlamp/train_log.json, outputs/SV_ULA/GM-LAMP-spread_gmlamp/train_log.json, outputs/SV_ULA/LISTA_gmlamp/train_log.json ... |
| S3-SV_UPA | main comparison SV_UPA (9 learned + NF-LADMM-S/C + AMP/OMP) + evaluation | todo (0/12) | outputs/SV_UPA/LAMP_gmlamp/train_log.json, outputs/SV_UPA/GM-LAMP-spread_gmlamp/train_log.json, outputs/SV_UPA/LISTA_gmlamp/train_log.json ... |
| S3-DeepMIMO_ULA | main comparison DeepMIMO_ULA (9 learned + NF-LADMM-S/C + AMP/OMP) + evaluation | todo (0/12) | outputs/DeepMIMO_ULA/LAMP_gmlamp/train_log.json, outputs/DeepMIMO_ULA/GM-LAMP-spread_gmlamp/train_log.json, outputs/DeepMIMO_ULA/LISTA_gmlamp/train_log.json ... |
| S3-DeepMIMO_UPA | main comparison DeepMIMO_UPA (9 learned + NF-LADMM-S/C + AMP/OMP) + evaluation | todo (0/12) | outputs/DeepMIMO_UPA/LAMP_gmlamp/train_log.json, outputs/DeepMIMO_UPA/GM-LAMP-spread_gmlamp/train_log.json, outputs/DeepMIMO_UPA/LISTA_gmlamp/train_log.json ... |
| S4-SV_ULA | seeds 1, 2 for 5 methods on SV_ULA + aggregate | todo (0/11) | outputs/SV_ULA/LAMP_gmlamp_s1/train_log.json, outputs/SV_ULA/LAMP_gmlamp_s2/train_log.json, outputs/SV_ULA/GM-LAMP-spread_gmlamp_s1/train_log.json ... |
| S4-SV_UPA | seeds 1, 2 for 5 methods on SV_UPA + aggregate | todo (0/11) | outputs/SV_UPA/LAMP_gmlamp_s1/train_log.json, outputs/SV_UPA/LAMP_gmlamp_s2/train_log.json, outputs/SV_UPA/GM-LAMP-spread_gmlamp_s1/train_log.json ... |
| S4-DeepMIMO_ULA | seeds 1, 2 for 5 methods on DeepMIMO_ULA + aggregate | todo (0/11) | outputs/DeepMIMO_ULA/LAMP_gmlamp_s1/train_log.json, outputs/DeepMIMO_ULA/LAMP_gmlamp_s2/train_log.json, outputs/DeepMIMO_ULA/GM-LAMP-spread_gmlamp_s1/train_log.json ... |
| S4-DeepMIMO_UPA | seeds 1, 2 for 5 methods on DeepMIMO_UPA + aggregate | todo (0/11) | outputs/DeepMIMO_UPA/LAMP_gmlamp_s1/train_log.json, outputs/DeepMIMO_UPA/LAMP_gmlamp_s2/train_log.json, outputs/DeepMIMO_UPA/GM-LAMP-spread_gmlamp_s1/train_log.json ... |
| S5-SV_UPA | scalable LADMM operators on SV_UPA | todo (0/4) | outputs/SV_UPA/LADMM-diag_gmlamp/train_log.json, outputs/SV_UPA/LADMM-lowrank_gmlamp/train_log.json, outputs/SV_UPA/LADMM-kron_gmlamp/train_log.json ... |
| S5-DeepMIMO_UPA | scalable LADMM operators on DeepMIMO_UPA | todo (0/4) | outputs/DeepMIMO_UPA/LADMM-diag_gmlamp/train_log.json, outputs/DeepMIMO_UPA/LADMM-lowrank_gmlamp/train_log.json, outputs/DeepMIMO_UPA/LADMM-kron_gmlamp/train_log.json ... |
| S6 | complexity: params / FLOPs / latency (script provided at this step) | todo (0/1) | outputs/S6_done.flag |
| S7 | robustness: new combiner, SNR mismatch (script provided at this step) | todo (0/1) | outputs/S7_done.flag |
| S8 | cross-dataset SV <-> DeepMIMO (script provided at this step) | todo (0/1) | outputs/S8_done.flag |
| S9 | per-channel NMSE CDF and sum-rate (script provided at this step) | todo (0/1) | outputs/S9_done.flag |
| S10 | ablations alpha=0, tied, positivity (script provided at this step) | todo (0/1) | outputs/S10_done.flag |

next step: S0b
