# bsce — PyTorch ports of all learned estimators, verified against the original code (Oct 2026)

Package: `bsce` (zip delivered in chat; README.md + VERIFICATION_REPORT.md inside).

## What exists
- Data: SV ULA/UPA channel generation (`generate_channel.m` order of random draws), CS matrix,
  `genMeasurements` and MATLAB test noise; scenario yaml files (GM-LAMP 2-band, LADMM 4-band,
  single band; DeepMIMO placeholders waiting for the user's DeepMIMO channels).
- Models (PyTorch, GPU if available else CPU): LAMP, GM-LAMP, LADMM (full / diag / lowrank / kron
  = the user's scripts; psd / kronprior / diagprior = proposed exact-ADMM variants), LISTA, ALISTA,
  LVAMP, ADMM-Net. Schedules: Borgerding/GM-LAMP layer-wise (warm start, round(nmse,5)), ALISTA,
  end-to-end. TF-1 Adam reproduced exactly (`adam_tf1`).
- Every trained model saved as .pt, .mat (matlab/bsce_infer.m) and, for LAMP/GM-LAMP/LADMM,
  `_orig.mat` that the unchanged LAMP.m / GM_LAMP.m / LADMM_test.m load.

## Verification (worst relative error, float64)
- data 8.8e-16 (CS matrix, genMeasurements bit-identical); MATLAB inference with shipped weights
  4.6e-15; TF graphs (outputs, gradients, Adam, every stage) 1.7e-13; LISTA/ALISTA/LVAMP 3.7e-14;
  original do_training loops replayed with the same minibatches: parameters after every stage
  1.3e-15 (LAMP, LADMM full/kron/diag), 4e-14 (GM-LAMP spread init); exports 1.4e-15.
- float32 originals (LADMM_train.py, ALISTA repo, LVAMP, THADMM-Net): 1e-5; float32 training loop
  agrees statistically (final NMSE 0.012 dB apart).

## Findings to remember
1. GM-LAMP's initial value has K identical mixture components -> the separating gradient is zero
   in exact arithmetic; components split only by round-off amplified by Adam (×10 per ~8-10 Adam steps).
   Not reproducible across devices; in PyTorch-CPU the components can stay identical.
   Recommended for experiments: `configs/train/gmlamp_spread.yaml` (means/log-vars offset 1e-3).
2. TF-1 Adam keeps hyper-parameters and beta powers in float32 even in float64 graphs.
3. Kronecker LADMM script computes (H2 ⊗ H1)x (row-major), docstring says H1 ⊗ H2.
4. LADMM alpha_{T-1} never gets a gradient.
5. round(nmse,5) validation -> ties at small NMSE -> earlier early stopping.
6. LADMM_UPA_test.m: T set before load(); weight files with T override it.
7. DeepMIMO CSmatrix stored single -> MATLAB evaluation in single precision.
8. genMeasurements leaves zero columns if K not divisible by #SNR chunks.
9. Uploaded DeepMIMO LADMM weights give ~0 dB (ULA) / -1 dB (UPA) NMSE on the GM-LAMP DeepMIMO
   test set + CSmatrix_128256 -> trained with other A/data; regenerate DeepMIMO results.

## Next
- User runs full training on GPU (`scripts/run_all.sh`, `tests/test_golden.py` with BSCE_DEVICE=cuda).
- DeepMIMO channels to be uploaded -> `generate_data.py --measurements-only`.
