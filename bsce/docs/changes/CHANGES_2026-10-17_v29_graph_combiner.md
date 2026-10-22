# v29 (17 Oct 2026): Graph-LADMM, and networks that work with any combiner and data set

Every default is unchanged: training, folder names, checkpoint fingerprints and `--methods all`. Runs of v28 resume and
evaluate as before.

## New model: Graph-LADMM (`bsce/models/graph_ladmm.py`)
It is unrolled ADMM on the whitened folded model y = A(x+n), with a magnetic beam graph:
- **nodes:** the beams;
- **edges:** the axis neighbours, with the wrap-around edge (ULA ring, UPA torus);
- **edge phase:** the leakage phase step between neighbouring beams, estimated from the training channels.

| option | values (default first) | what |
|---|---|---|
| `model.zstep` | `gtv_gate` (gladmm), `gate` (gladmmf), `gtv_gm`, `gtv`, `gm`, `gsoft`, `l1`, `gm_graph`, `gtv_gm_graph` | the node step; `gtv*` add the graph-TV split z₂ = Dx |
| `model.xstep` | `analytic`, `free` | (P+ρ(I+DᴴD))⁻¹ through the combiner (any combiner), or a free H_t per layer (one combiner) |
| `model.sigma` | `residual`, `blind`, `known` | the noise level of the thresholds |
| `model.edge_phase` | `data`, `theory`, `blind`, degrees | the phases come from the training channels, −π/n, 0, or a fixed value |
| `model.learn_phase`, `model.ring`, `model.gm_K`, `model.lr_mult`, `model.graph_pool` | false, true, 4, 10, true | train the phases; wrap edges; GM components; scalar learning-rate factor; graph pooling of the gate (false = ablation) |

**Method keys (not in `all`):**
- `gladmm`: analytic, T = 8;
- `gladmmf`: free H_t, T = 6;
- `gladmmany`: analytic, trained on random combiners, T = 8.

Configs `gladmm.yaml`, `gladmm_free.yaml` and `gladmm_anyA.yaml`. Their own training is `--protocol paper`: end to end, online noise, log-NSE loss.

**Measured edge phases** (training channels):

| data set | phase | coherence |
|---|---|---|
| SV ULA, SV UPA | ±180° interior, 0° wrap (centred array index) | 0.87–0.97 |
| DeepMIMO ULA | +1.3° | 0.92 |
| DeepMIMO UPA | +11.2° | 0.93–0.96 |
| 3GPP ULA | 0.2° | 0.35 |
| 3GPP UPA | 11–12° | 0.37–0.99 |

## Combiner
- **`set_combiner(A)`:** Graph-LADMM analytic and LADMM `operator=analytic|analytic1` follow a new combiner. Every other network refuses: it was learned for its combiner.
- **`train.combiner=random`:** a new random ±1/√M combiner every minibatch. It needs online noise. Validation, checkpoints and the saved network use the scenario's combiner. `finetune_bands.py --combiner auto|fixed|random`: auto keeps the source's choice.
- **`evaluate.py --combiner <file>|random:<seed>`:** tests with another combiner.
  - A runs folder of another scenario is tested on this scenario's combiner the same way. DeepMIMO's combiner is not SV's.
  - `eval_log.json` lists which networks followed and which kept their own combiner.
  - `--graph-phase target`: analytic Graph-LADMM takes the target data set's edge phases. With a free H_t it keeps
    its own, because H_t was learned with them.
- **Workbench:** step 5, "Another combiner or data set".

## Checks
`tests/test_v29_graph_combiner.py`:
- the port equals the graph-choice study's layer (≤ 1e-12, every z-step, ULA and UPA);
- zstep `gtv` is ADMM for the fused LASSO (an independent numpy ADMM, ≤ 1e-10);
- free H at init gives the analytic step;
- the phase estimator gives 180° on SV;
- `set_combiner` equals a network built on the new combiner;
- the random-combiner training leaves the scenario's combiner in the network and its checkpoint;
- the evaluation on another combiner and on DeepMIMO;
- the graph-coupled GM starts as the plain GM.

The CPU search of the configurations is in the v29 report.
