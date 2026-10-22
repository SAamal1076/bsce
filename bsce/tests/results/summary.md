| Check | Group | n | worst relative error | |
|---|---|---|---|
| SV channels x = U.'h (vs generate_channel.m) | ULA | 5 | x_rel=5.6e-16 |
| SV channels x = U.'h (vs generate_channel.m) | UPA | 5 | x_rel=8.8e-16 |
| Full authors' data script, ULA, S=80000 |  | 1 | rel=5.4e-16 |
| LADMM-project data script (100k + 10k channels per array) | ULA | 2 | rel=5.5e-16 |
| LADMM-project data script (100k + 10k channels per array) | UPA | 2 | rel=8.7e-16 |
| CS matrix (generate_CS_matrix.m) |  | 3 | rel=0.0e+00 |
| Measurements, MATLAB noise (Main.m) |  | 6 | y_rel=5.3e-16 |
| Measurements, genMeasurements (python) |  | 9 | rel=0.0e+00 |
| LAMP / GM-LAMP vs LAMP.m / GM_LAMP.m (trained weights) | GM-LAMP | 32 | rel=4.6e-15 |
| LAMP / GM-LAMP vs LAMP.m / GM_LAMP.m (trained weights) | LAMP | 32 | rel=1.2e-15 |
| LAMP / GM-LAMP vs TF graph, all outputs (T=8) | gm | 4 | fwd_rel=3.8e-16 |
| LAMP / GM-LAMP vs TF graph, all outputs (T=8) | soft | 4 | fwd_rel=6.1e-16 |
| LAMP / GM-LAMP vs TF setup_training (grads, Adam) | gm | 3 | fwd_rel=3.7e-16, grad_rel=8.1e-15, adam_rel=4.9e-14 |
| LAMP / GM-LAMP vs TF setup_training (grads, Adam) | soft | 3 | fwd_rel=6.1e-16, grad_rel=4.0e-16, adam_rel=5.0e-15 |
| LADMM vs LADMM_test.m / LADMM_UPA_test.m (trained weights) | ULA | 8 | rel=2.0e-15, layer_nmse_rel=2.6e-15 |
| LADMM vs LADMM_test.m / LADMM_UPA_test.m (trained weights) | UPA | 8 | rel=1.4e-15, layer_nmse_rel=1.7e-15 |
| LADMM (analytic init) vs ADMM_vectorized.m |  | 3 | rel=4.4e-15 |
| LADMM scripts (full/diag/lowrank/kron) vs TF graphs | diag (float64) | 2 | init_rel=0.0e+00, fwd_rel=9.8e-16, grad_rel=4.5e-15, adam_rel=1.7e-13 |
| LADMM scripts (full/diag/lowrank/kron) vs TF graphs | full (float32) | 2 | init_rel=0.0e+00, fwd_rel=3.1e-07, grad_rel=4.4e-06, adam_rel=1.0e-05 |
| LADMM scripts (full/diag/lowrank/kron) vs TF graphs | full (float64) | 2 | init_rel=0.0e+00, fwd_rel=1.1e-15, grad_rel=1.0e-14, adam_rel=2.6e-14 |
| LADMM scripts (full/diag/lowrank/kron) vs TF graphs | kron (float64) | 2 | init_rel=0.0e+00, fwd_rel=4.4e-16, grad_rel=1.6e-14, adam_rel=1.7e-13 |
| LADMM scripts (full/diag/lowrank/kron) vs TF graphs | lowrank (float64) | 2 | init_rel=0.0e+00, fwd_rel=9.7e-16, grad_rel=3.0e-14, adam_rel=4.7e-15 |
| LISTA / ALISTA vs ALISTA-repo TF | ALISTA (float32) | 2 | fwd_rel=0.0e+00, grad_rel=1.7e-07, adam_rel=0.0e+00 |
| LISTA / ALISTA vs ALISTA-repo TF | ALISTA (float64) | 2 | fwd_rel=4.7e-15, grad_rel=1.8e-15, adam_rel=0.0e+00 |
| LISTA / ALISTA vs ALISTA-repo TF | LISTA (float32) | 2 | fwd_rel=0.0e+00, grad_rel=2.7e-06, adam_rel=7.7e-06 |
| LISTA / ALISTA vs ALISTA-repo TF | LISTA (float64) | 2 | fwd_rel=4.7e-16, grad_rel=2.3e-15, adam_rel=3.7e-14 |
| LVAMP vs Borgerding TF | float32 | 3 | fwd_rel=4.6e-07, grad_rel=5.8e-06, adam_rel=4.8e-07 |
| LVAMP vs Borgerding TF | float64 | 3 | fwd_rel=9.7e-16, grad_rel=1.0e-14, adam_rel=1.8e-15 |
| ALISTA analytic W vs CalculateW.m |  | 3 | rel=0.0e+00 |
| ADMM-Net vs THADMM-Net repo (PyTorch) |  | 3 | fwd_rel=0.0e+00, grad_rel=4.7e-08, adam_rel=2.5e-08 |
| AMP / OMP / SD / ADMM ports vs .m files |  | 4 | amp_rel=9.8e-16, omp_rel=4.9e-16, sd_rel=5.0e-16, admm_rel=2.9e-14 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | ADMM-Net | 2 | mat_reload_rel=0.0e+00, matlab_rel=1.4e-15 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | ALISTA | 2 | mat_reload_rel=0.0e+00, matlab_rel=7.1e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | GM-LAMP | 2 | mat_reload_rel=0.0e+00, matlab_rel=4.3e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-diag | 2 | mat_reload_rel=0.0e+00, matlab_rel=1.1e-15 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-diagprior | 2 | mat_reload_rel=0.0e+00, matlab_rel=2.0e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-full | 2 | mat_reload_rel=0.0e+00, matlab_rel=2.1e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-kron | 2 | mat_reload_rel=0.0e+00, matlab_rel=8.0e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-kronprior | 2 | mat_reload_rel=0.0e+00, matlab_rel=2.1e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-lowrank | 2 | mat_reload_rel=0.0e+00, matlab_rel=1.3e-15 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LADMM-psd | 2 | mat_reload_rel=0.0e+00, matlab_rel=1.7e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LAMP | 2 | mat_reload_rel=0.0e+00, matlab_rel=4.0e-16 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LISTA | 2 | mat_reload_rel=0.0e+00, matlab_rel=5.6e-17 |
| Export .mat -> MATLAB bsce_infer.m vs PyTorch | LVAMP | 2 | mat_reload_rel=0.0e+00, matlab_rel=4.7e-16 |
| Export _orig.mat -> ORIGINAL LAMP.m/GM_LAMP.m/LADMM_test.m | GM-LAMP | 2 | rel=4.5e-16 |
| Export _orig.mat -> ORIGINAL LAMP.m/GM_LAMP.m/LADMM_test.m | LADMM-full | 2 | rel=3.9e-16 |
| Export _orig.mat -> ORIGINAL LAMP.m/GM_LAMP.m/LADMM_test.m | LADMM-kron | 2 | rel=7.5e-16 |
| Export _orig.mat -> ORIGINAL LAMP.m/GM_LAMP.m/LADMM_test.m | LADMM-kronprior | 2 | rel=3.0e-16 |
| Export _orig.mat -> ORIGINAL LAMP.m/GM_LAMP.m/LADMM_test.m | LAMP | 2 | rel=3.5e-16 |
| Proposed LADMM (init) vs classical ADMM | diagprior | 2 | rel=4.8e-15 |
| Proposed LADMM (init) vs classical ADMM | kronprior | 2 | rel=4.8e-15 |
| Proposed LADMM (init) vs classical ADMM | psd | 2 | rel=4.8e-15 |
| Complex LISTA/ALISTA on real data == real version | ALISTA | 1 | rel=2.3e-15 |
| Complex LISTA/ALISTA on real data == real version | LISTA | 1 | rel=1.1e-16 |
| Original do_training loop, params after every stage (exact part) | GM-LAMP original init (float64) | 1 | exact_param_rel=5.6e-16, exact_dnmse_db=5.0e-07 |
| Original do_training loop, params after every stage (exact part) | GM-LAMP spread init (float64) | 1 | exact_param_rel=4.0e-14, exact_dnmse_db=4.7e-07 |
| Original do_training loop, params after every stage (exact part) | LADMM-diag (float64) | 1 | exact_param_rel=3.7e-16, exact_dnmse_db=4.8e-07 |
| Original do_training loop, params after every stage (exact part) | LADMM-full (float64) | 1 | exact_param_rel=1.2e-15, exact_dnmse_db=4.8e-07 |
| Original do_training loop, params after every stage (exact part) | LADMM-kron (float64) | 1 | exact_param_rel=2.6e-16, exact_dnmse_db=4.3e-07 |
| Original do_training loop, params after every stage (exact part) | LAMP (float64) | 1 | exact_param_rel=1.3e-15, exact_dnmse_db=4.7e-07 |
| LADMM_train.py loop (float32): final NMSE difference [dB] | LADMM-full float32 (float32) | 1 | final_dnmse_db=1.2e-02 |
