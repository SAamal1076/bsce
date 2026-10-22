# Stand-alone audit scripts (docs/AUTHORS_PROTOCOL.md)

They use only NumPy/SciPy and the authors' package (no bsce code), with a NumPy port of the authors' TensorFlow
network (`authors_forward.py`). `O` = the folder of the authors' package, `D` = the bsce `data/` folder.

| script | what | command |
|---|---|---|
| `a1_npz_vs_mat.py` | the 16 `.npz` training states hold the weights of the 16 published `.mat` networks | `python a1_npz_vs_mat.py O` |
| `a2_published_on_our_val.py` | the published networks on our validation sets vs the authors' logged validation NMSE (bootstrap sd, z); and on the test sets | `python a2_published_on_our_val.py O D out/` |
| `a3_measurements.py` | the measurement files: SNR re-estimated from y − Ax with the authors' A, chunks, band | `python a3_measurements.py O D out/` |
| `a4_val_percentile.py` | the authors' validation NMSE and the test NMSE among 5000 random 2000-channel sets of our data | `python a4_val_percentile.py O D out/` (about 15 min) |
| `a5_ula_val.py` | DeepMIMO ULA: our validation users with fresh noise vs random sets | `python a5_ula_val.py O D DeepMIMO_ULA` |
| `plot_percentiles.py` | the figure of a4 | `python plot_percentiles.py out/validation_percentiles.json out/validation_percentiles.png` |

`scripts/check_vs_authors.py` combines a2 and a3, and compares the training stages, inside bsce.
