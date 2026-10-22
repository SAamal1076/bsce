# v31.1 (18 Oct 2026): GM-LAMP trains as in the authors' code again

**In short.** GM-LAMP trained with bsce from the original init stayed near the level of a linear estimator. Its 4 mixture components
never separated. This held on every device and at both precisions, for v31 and every earlier version.
- **v31.1 separates them** before training.
- **Retrain** every GM-LAMP network trained with bsce before v31.1 from the original init. `scripts/check_gm_components.py`
  lists them.
- The authors' published networks, networks trained with `gmlamp_spread.yaml` and every other method are unaffected.

## 1. What was wrong
- **The original init.** GM-LAMP's init (`gm_init: original`, the authors' `define_shrinkage.py`) gives the K = 4
  components of every layer the same logit, mean and log-variance.
- **The gradients.** With identical components, PyTorch computes bit-identical gradients for them:
  - measured on the CPU in float32 and float64 (`gm_symmetry_evidence.py` of the v31.1 report, `--device cuda` on a GPU);
  - the GPU log of 18 Oct shows the same plateau;
  - the logits get exactly 0.

  Adam then moves every component by the same amount, so they stay identical for the whole training.
- **The result.** The mixture is one Gaussian. That denoiser scales every entry by the same factor (which depends on the
  residual), so it cannot use the sparsity. The network stays near the level of a linear estimator.
  - SV ULA: −2.93 dB from layer 1 to layer 4.
  - Then −4.5 dB at layer 5, only through that residual-dependent factor (your GPU log).
  - The authors' log: −13.9 dB at layer 5.
- **Why the authors' training is fine.** In exact arithmetic the separating gradient is zero there too. TensorFlow's
  kernels add round-off that differs between the components, and training amplifies it. That was the known reason for
  `gm_init: spread` (`gm_theta_init` docstring). The default path did not use spread, though:
  - `gmlamp_original.yaml`;
  - `run_suite.py --methods gmlamp`, any protocol;
  - Spyder cell 2.2;
  - the workbench's GM-LAMP rows.

**What it looked like** (SV ULA, 0–10 dB band, validation at 10 dB):

| stage (validation NMSE, iterations) | authors' log | bsce ≤ v31 (the GPU log of 18 Oct) | v31.1 (CPU) |
|---|---|---|---|
| linear T=1 | −2.9457 (14k) | −2.8965 (14k) | −2.8965 (14k) |
| non-linear T=1 | −3.2443 (21k) | −2.9312 (36k) | −3.0290 (22k) |
| non-linear T=1 trainrate=0 | −9.1786 (9k) | −2.9111 (6k) | **−9.0309** (9k) |
| layer 5 | −13.92 | −4.56 after 35 h, still in its first refinement | (retrain) |

The first four stages, where the mixture does not enter, are identical on the GPU and the CPU (all 47 printed values).

The v30 report's row "GM-LAMP, end to end, 3,000 steps: −3.73 dB (fails)" was this bug. All 8 layers of that network
have identical components.

With v31.1 the same command gives **−9.96 dB** (mean over 0–20 dB on the authors' test set; −6.01 / −10.68 / −11.80 dB
at 0 / 10 / 20 dB).
- That is +6.2 dB, but still well below:
  - the authors' layer-wise networks: −16.87 dB;
  - LAMP with the same 3,000 end-to-end steps: −12.94 dB.
- The original prior (every variance about 1) adapts slowly. 3,000 end-to-end steps are too few for GM-LAMP; its
  validation was still improving at the end. Use the layer-wise protocol for a GM-LAMP baseline.

## 2. The fix
- **`LAMP.break_gm_symmetry(names)`.** For every `theta_t` in `names` whose K components are still identical, it adds
  the `gm_init: spread` offset: 10⁻³·linspace(−1, 1, K) on the means and the log-variances. The values are then those
  of `gm_theta_init(K, "spread")`. Components that already differ are left alone, and the call returns the names it
  changed.
- **The trainer.**
  - It calls the method before a stage trains such a θ (`train.gm_symmetry: break`, the default).
  - It prints a note and records the names in the training state (`gm_symmetry_broken`).
  - `train.gm_symmetry: keep` gives the original arithmetic; the TF replay tests use it.
  - A resumed stage is left as saved.
- **Old checkpoints are not resumed.** Take an interrupted GM-LAMP training whose checkpoint was written before v31.1,
  and whose trained mixtures still have identical components. That checkpoint is set aside as `<name>.pre_v31_1.pt`,
  and the training starts again. So the same command retrains, and gives exactly the network of a fresh v31.1 run.
  - A run that had finished (its `.pt` files exist) is still skipped by `run_suite.py`.
  - Move such folders away first; `check_gm_components.py` lists them.
- **`finetune_bands.py`.** It warns when it starts from an affected network, and gives it the same offset, so the
  fine-tune can separate the components. A retrained network is better.
- **`scripts/check_gm_components.py --runs "outputs/*"`** lists the affected networks. Exit code 1 when it finds one.

## 3. Checks
- **`tests/test_v31_release.py`, 5 new tests:**
  - the gradient is identical across the components (float32 and float64);
  - the break equals the spread init, and a second call changes nothing;
  - with `break` the training separates the components; with `keep` they stay identical;
  - an old checkpoint is set aside, and the run then equals a fresh v31.1 run bit for bit. A v31.1 checkpoint resumes;
  - `check_gm_components.py`: affected network → exit 1; spread and soft LAMP → exit 0; old checkpoints listed apart.
- **`test_training_loop_vs_tf.py`:** the original-init replay of the TF loop uses `gm_symmetry: keep` and is unchanged.
- **`tests/test_gpu_vs_cpu.py::test_gmlamp_original_init_on_this_device`** (new) runs on the GPU machine:
  - it records whether the components get identical gradients there (`-s` prints it);
  - it checks that the trainer separates them on that device.

  Command: `python -m pytest tests/test_gpu_vs_cpu.py -k original_init -s`.
- **The layer-1 runs above** (CPU, float32, the full stage list of `gmlamp_original.yaml`).
- **The v30 end-to-end command rerun.**
- **`check_gm_components.py`:**
  - affected: the v30 end-to-end network and a v31 original-init network;
  - not affected: the spread and the authors' networks.
