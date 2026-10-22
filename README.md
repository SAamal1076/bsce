# bsce

Learned beamspace channel estimation for mmWave massive MIMO in PyTorch
(LAMP, GM-LAMP, the LADMM family, Graph-LADMM, NF-LADMM, LISTA, ALISTA, LVAMP, ADMM-Net, CADMM, UTRR and the
classical baselines), each port verified against the original MATLAB / TensorFlow code.

**Start with [bsce/README.md](bsce/README.md)** (installation, quick start) and [bsce/GUIDE.md](bsce/GUIDE.md)
(the workflow).

## Versions in the history

| commit | what |
|---|---|
| `Add bsce v31.2` | the package as delivered (19 Oct 2026), unchanged |
| `Fix the bugs of review round 1 (R1-R14)` | training, tables and export fixes |
| `bsce v31.3` | new methods `ladmmat`, `ladmmatr`, `ladmmatrd` (noise-adaptive LADMM) and the round 2 fixes |

What changed and why: [bsce/docs/REVIEW_2026-10-21.md](bsce/docs/REVIEW_2026-10-21.md) (both review rounds) and
[bsce/docs/changes/CHANGES_2026-10-22_v31_3.md](bsce/docs/changes/CHANGES_2026-10-22_v31_3.md) (v31.3 and its pilot
results).

## Not included

The reference zips (`bsce_reference_part1.zip` to `part4.zip`: the GM-LAMP authors' CS matrices, test files and
trained weights, and the original code of LADMM, Borgerding, ALISTA and THADMM-Net) are third-party files and are
not in this repository. Extract them next to the code as described in `bsce/README.md` section 1. Data sets
(`bsce/data/`), trained networks (`bsce/outputs/`) and results (`bsce/results/`) are made by the scripts.
