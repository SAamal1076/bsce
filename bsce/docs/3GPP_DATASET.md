# 3GPP TR 38.901 data sets (UMi street canyon, 28 GHz), configured like the SV and DeepMIMO data

**In one command** (Spyder, bsce environment, in the bsce folder; about 2 min per array):
```
%run scripts/make_3gpp_data.py
```
- **What it makes:** `data/3GPP_UPA` and `data/3GPP_ULA`, 80,000 / 2,000 / 2,000 channels (train / val / test), in the format of `data/SV_*` and `data/DeepMIMO_*`.
- **Scenario files:** `configs/scenarios/3gpp_upa.yaml` and `3gpp_ula.yaml` (shipped).
- **Training:** train them as the other scenarios:
  ```
  %run scripts/run_suite.py --scenarios all 3gpp --methods ladmm6 ladmm4 lamp8 gmlamp8 nfladmmsr4 --device cuda
  ```
  (`all` = the four GM-LAMP sets, `3gpp` = the two new ones).

**What the data are.**
- **The channel model:** 3GPP TR 38.901 §7.5 fast fading, UMi street canyon, with LOS/NLOS drawn per user by §7.4.2 and the BS element of §7.3.
- **The rest is as in bsce's DeepMIMO data:** the 28 GHz carrier, the BS arrays, narrowband channels, the lens, the normalisation, the CS matrix and the SNR bands.
- **The generator:** bsce's own Python implementation of the TR procedure, `bsce/data/tr38901.py`.
  - It runs in the bsce environment. No MATLAB, no Sionna, no extra package.
  - It agrees with Sionna 2.2, an independent implementation of 38.901, within the sampling spread (section 5).

**Why not MATLAB?** You have MATLAB, so this was a real option. The Python generator is the better choice for these reasons:
- **Speed:** it is about 80 times faster than QuaDRiGa, about 1.5 min against about 2 h per 100,000 channels.
- **Format:** it writes bsce's data-set format directly.
- **Fidelity:** it follows the TR's equations. QuaDRiGa 2.8.1 deviates from 38.901 in four places (section 5.2).

`matlab/quadriga_make_H.m` (QuaDRiGa) and `scripts/sionna_make_H.py` (Sionna) stay available as second implementations (section 3.6).

---

## 1. How many examples?

**The standard defines none.** TR 38.901 is a channel model. Every channel is a new random draw, and the data-set size is the user's choice.

The data-set sizes of related papers:

| paper | channels | train / val / test | bsce preset |
|---|---|---|---|
| Wei, Hu, Dai, "Deep learning for beamspace channel estimation in mmWave massive MIMO", IEEE TCOM (arXiv 1910.12455): the protocol GM-LAMP follows | Saleh–Valenzuela | 80,000 / 2,000 / 2,000 | `sv` (**default**) |
| the same paper | DeepMIMO O1, 28 GHz | 50,000 / 2,000 / 2,000 | `deepmimo` |
| Dong, Zhang, Li, Gaspar, NaderiAlizadeh, "Deep CNN-based channel estimation for mmWave massive MIMO systems", IEEE JSTSP 2019 (arXiv 1904.06761) | 3GPP TR 38.901 (CDL), 28 GHz | 81,000 / 9,000 / 19,000 | `dong2019` |
| 100,000 channels per array | 3GPP | 96,000 / 2,000 / 2,000 | `100k` |

- **The default is `sv`**, so that the three data sets have the same size as the SV data, and every network sees the same number of training examples.
- `dong2019` gives the split of the 3GPP paper.
  - Dong et al. used the CDL (clustered delay line) link-level profiles of 38.901 with wideband channels.
  - bsce uses the full stochastic UMi model of §7.5 with per-user large-scale parameters, narrowband, as its SV and DeepMIMO data.
  - Only the sizes are taken from the paper.
- **Any size:** `--n-train 120000 --n-val 5000 --n-test 5000` (give all three).
- **The validation and test sets do not depend on the training size.**
  - They come from their own drops: drop 1,000,001 onward for validation and 2,000,001 onward for test, while training uses drops 1, 2, …
  - So `--sizes sv` and `--sizes 100k` have the same 2,000 validation and 2,000 test users, and a larger training set only adds users.
- **The three arrays see the same users,** because every drop has a fixed seed.

---

## 2. The configuration, item by item

| item | SV (bsce = GM-LAMP) | DeepMIMO (bsce = GM-LAMP) | 3GPP (this guide) | note |
|---|---|---|---|---|
| channel model | Saleh–Valenzuela: 3 paths, gains CN(0, 1), angles uniform | ray tracing, O1_28, BS 3, rows 1000–1300; the 3 strongest paths | 38.901 §7.5, UMi street canyon: 12 clusters (LOS) or 19 (NLOS) × 20 rays; large-scale parameters of Table 7.5-6 (v16.1; v19.2 optional) | the standard, unchanged |
| carrier | none (angles only, λ/2) | 28 GHz | **28 GHz** | = DeepMIMO; the 38.901 tables depend on fc |
| bandwidth | narrowband | one subcarrier | **one frequency, the carrier**: h = Σ ray coefficients | same |
| BS array | ULA 256 / UPA 16×16, λ/2 | ULA 256 on z / UPA 16×16 on y–z, λ/2 | **UPA 16×16 on y–z; ULA 256 on y** (or on z), λ/2, boresight +x | same arrays; the ULA axis is a choice (section 4) |
| BS element | isotropic | half-wave dipole (DeepMIMO v1.3) | 38.901 Table 7.3-1 element (65°, 8 dBi, 30 dB front-back), V-polarised; `--element omni` = isotropic | the standard's element |
| UE | n/a | one antenna | one isotropic antenna, height 1.5 m | same |
| users | n/a | outdoor users of a 60 m street strip, on a 0.2 m grid | **uniform in the 120° hexagonal sector cell** of one site: ISD 200 m, 10 m ≤ d2D ≤ 133 m, outdoor; BS at 10 m | the 38.901 UMi layout (Table 7.8-1), one sector |
| LOS / NLOS | n/a | from the geometry | per user from Table 7.4.2-1 (`--channel umi`); LOS-only / NLOS-only also possible | the standard |
| wavefront | plane | plane | plane (38.901 baseline) | same |
| path loss, shadowing | none | in h, removed by the normalisation | not generated (removed by the normalisation anyway) | same |
| normalisation | E‖h‖² = N (on average) | ‖x‖² = N per channel | **‖x‖² = N per channel** | SNR = 1/σ² per channel, as DeepMIMO |
| beamspace | x = Uᵀh (SV lens matrix) | x = U0ᴴh | **x = U0ᴴh** (UPA: U0 ⊗ U0) | the DeepMIMO convention |
| CS matrix (M = 128) | `CSmatrix256128` | `CSmatrix_128256` | `CSmatrix256128` | same kind and size |
| measurements | y = A(x + n) | the same | the same | same |
| SNR bands (authors' protocol) | 0–10 / 10–20 dB, 10 chunks, validation at 10 dB | the same | the same; `--snr` for others | same |
| sizes (train / val / test) | 80k / 2k / 2k | 50k / 2k / 2k | **80k / 2k / 2k** (`--sizes`, section 1) | |
| test set | independent draws | GM-LAMP's test file: users interleaved with training users on the 0.2 m grid | **held-out drops**: validation and test users in drops no training user belongs to | stricter than DeepMIMO |

Two things differ by nature, not by configuration: the number of propagation paths and the user geometry. They are
what this data set adds.

---

## 3. Make it

### 3.1 One command: `scripts/make_3gpp_data.py`
```
%run scripts/make_3gpp_data.py                                  3GPP_UPA + 3GPP_ULA, 80,000 / 2,000 / 2,000
%run scripts/make_3gpp_data.py --sizes dong2019                 81,000 / 9,000 / 19,000
%run scripts/make_3gpp_data.py --n-train 96000 --n-val 2000 --n-test 2000
%run scripts/make_3gpp_data.py --only 3GPP_ULAZ                 the vertical ULA as well (data/3GPP_ULAZ)
%run scripts/make_3gpp_data.py --spec 19.2 --suffix _v19        a second copy with the Rel-19 tables
%run scripts/make_3gpp_data.py --check                          and check them afterwards (section 3.3)
```

| option | default | what it does |
|---|---|---|
| `--sizes sv\|deepmimo\|dong2019\|100k` | `sv` | the (train, val, test) preset of section 1 |
| `--n-train N --n-val N --n-test N` | | exact sizes instead (all three) |
| `--only` | `3GPP_UPA,3GPP_ULA` | which sets: `3GPP_UPA` (UPA 16×16), `3GPP_ULA` (ULA 256 on y), `3GPP_ULAZ` (ULA 256 on z) |
| `--spec 16.1\|19.2` | `16.1` | the 38.901 tables (v16.1, or the Rel-19 update) |
| `--channel umi\|umi_los\|umi_nlos` | `umi` | LOS/NLOS by Table 7.4.2-1, or all LOS / all NLOS |
| `--element 38.901\|omni` | `38.901` | the BS element of Table 7.3-1, or isotropic |
| `--snr <edges>` | the authors' bands | e.g. `--snr 0 20` for one band; `train.py` makes missing band files itself |
| `--suffix` | | e.g. `_v19`: `data/3GPP_UPA_v19` and `configs/scenarios/3gpp_upa_v19.yaml` |
| `--seed` | 2026 | another seed gives another data set with the same statistics |
| `--drop-users` | 250 | users per drop (the spatial correlation of the LSPs is within a drop) |
| `--out-root` | `data` | the parent folder of the data sets |
| `--check` | off | `check_3gpp.py --builtin` on 10,000 channels afterwards |
| `--force` | off | make a set again (an existing set is kept otherwise) |
| `--dry-run` | off | print the commands only |

**What `data/3GPP_UPA` holds** (the same as `data/SV_UPA`, plus a meta file):

| file | content |
|---|---|
| `3GPP_UPA_{training,validation,test}_channel.mat` | `x`, 256 × K complex beamspace channels, ‖x‖² = 256 each |
| `3GPP_UPA_{training,validation}_128_measurements_{0to10dB,10to20dB}.mat` (+ `.snr.json`) | `y` (128 × K), `snr_db`: the authors' bands, 10 SNR chunks in training, 10 dB in validation |
| `CSmatrix.mat` | `A`, 128 × 256: SV's `CSmatrix256128` |
| `3GPP_UPA_meta.mat` | per set: `<set>_loc` (x, y, z in m), `_los`, `_drop`, `_lsp` (DS s, ASD, ASA, ZSD, ZSA deg, K dB with −100 for NLOS, SF dB) |
| `dataset_info.json` | sizes, generator settings, LOS fraction per set, drops shared between sets (0) |
| `dataset_check.png` | channel energy, strongest-beam histogram (train vs test), SNR of the measurement files |

The test measurements are made by `evaluate.py` at each test SNR, as for SV and DeepMIMO.

**Time** (measured here on 2 CPU cores):
- About 0.9 s per 1,000 UPA channels and 1 s per 1,000 ULA channels.
- `make_3gpp_data.py --check` (two sets of 84,000 channels, the measurement files, the check): 4 min 16 s.

### 3.2 Directly with `scripts/make_dataset.py`
`make_3gpp_data.py` calls this, one set at a time. With no `--input`, `--dataset 3GPP` uses the built-in generator:
```
%run -G scripts/make_dataset.py --dataset 3GPP --array UPA --sizes sv --authors-bands --name 3GPP_UPA_80k
%run -G scripts/make_dataset.py --dataset 3GPP --array ULA --ula-axis y --n-train 96000 --n-val 2000 --n-test 2000 --authors-bands --name 3GPP_ULA_100k
```
- **Options of the built-in path:** `--ula-axis y|z`, `--spec`, `--channel`, `--element`, `--drop-users`, `--seed`.
- **`--raw-out data/raw3gpp/H_upa`** also writes the raw channels H (before the lens), 5,000 per file, in the format of the other generators.
- **`--test random`:** validation and test users drawn at random from one pool (as `--test random` of SV/DeepMIMO), instead of held-out drops.

### 3.3 Check
```
%run -G scripts/check_3gpp.py --builtin UPA ULA_Y ULA --n 10000
```
- **Output:** `results/check_3gpp/`: `check_3gpp.md`, `.json` and six figures (section 5).
- **Needs:** bsce's SV and DeepMIMO data (`data/SV_*`, `data/DeepMIMO_*`, step 1 of `GUIDE.md`) for the comparison with them.
- **What to look for:**
  - `lsp_vs_38901.png`: every "median offset" within ±0.03 decades, and every log-sd near the table's.
  - `los_k_vs_38901.png`: the LOS points on the 38.901 curve, and the K-factor on N(9 dB, 5 dB).
  - `beamspace_3datasets.png`, `lmmse_3datasets.png`, `transfer_gmlamp.png`: how the data compare with SV and DeepMIMO.
- **Other generators:** `--raw UPA="data/raw3gpp/H_upa_part*.mat"` checks raw files of Sionna, QuaDRiGa or `--raw-out`.

### 3.4 Train and compare
Train with the same commands as the other scenarios. The scenario names are `3gpp_upa` and `3gpp_ula`, and `3gpp` means both.

The default suite (`GUIDE.md` 4.1):
```
%run scripts/run_suite.py --scenarios all 3gpp --methods ladmm6 ladmm4 lamp8 gmlamp8 nfladmmsr4 --device cuda
%run scripts/compare.py --scenarios all 3gpp --name suite_6sets
```
The authors' protocol for every method (`GUIDE.md` 4.8):
```
%run scripts/run_suite.py --scenarios 3gpp --methods ladmm6 ladmm4 lamp8 gmlamp8 nfladmmsr4 --protocol gmlamp --train-snr scenario --ft-snr none --device cuda
%run scripts/compare.py --scenarios 3gpp --tags "*_gmlamp_trscenario*" --name 3gpp_authors_protocol
```
- **Reference for the gains:** the authors published no networks for these data, so the reference is a GM-LAMP trained here.
  - `compare.py` takes the first of `GM-LAMP-T8_ft_0-10-20dB_*`, `GM-LAMP-T8-ft`, `GM-LAMP-T8`, `GM-LAMP-T8_gmlamp_trscenario`, `GM-LAMP-T8_paper_trscenario` found in each table.
  - `--ref <tag>` chooses another.
- **Two computers:** run `--scenarios 3gpp_upa` on one and `--scenarios 3gpp_ula` on the other. Make the data on each computer with the same command; the data are the same (fixed seeds).

### 3.5 Workbench
- **Step 1:** "3GPP TR 38.901 data sets (built-in, Python only)". It has the size preset or exact sizes, the sets, the tables, LOS/NLOS, the element, a suffix, and the check.
- **Steps 2 to 4:** the scenarios `3GPP UPA` and `3GPP ULA` appear with the other four.
- **"Data set of any size":** offers 3GPP with the size presets and the ULA axis.

### 3.6 Second implementations: Sionna and QuaDRiGa (MATLAB)
These are for a robustness check, not needed for the data sets above. Both write raw files that `make_dataset.py --dataset 3GPP --input "<files>"` turns into a data set in the same format.

**Sionna 2.2** follows the TR as written, and is the implementation the built-in generator was checked against. It needs PyTorch ≥ 2.9.1, so it gets its own environment:
```
conda create -n sionna python=3.12 -y
conda activate sionna
pip install sionna-no-rt==2.2.0 scipy
python scripts/sionna_make_H.py --array UPA   --n 100000 --out data/raw3gpp/H_3gpp_upa
python scripts/sionna_make_H.py --array ULA_Y --n 100000 --out data/raw3gpp/H_3gpp_ulay
```
- **Speed:** 11–35 ms per channel, so 20–60 min per array.
- **Parts:** 20 part files of 5,000. A stopped run resumes, and `--parts 1-10` / `--parts 11-20` share the work between two computers.
- **Options:** `--spec 19.2`, `--scenario umi_los|umi_nlos`, `--element omni`, `--indoor 0.8`, `--device cuda`.

**QuaDRiGa 2.8.1** (Fraunhofer HHI) in MATLAB or Octave:
- **Get it:** github.com/fraunhoferhhi/QuaDRiGa, Code → Download ZIP. Keep the folder name `quadriga_src`.
- **Run, in MATLAB:**
  ```matlab
  addpath('D:\Academic Profile\LADMM-MIMO\codes\bsce\matlab')
  q = 'D:\QuaDRiGa\quadriga_src';
  o = 'D:\Academic Profile\LADMM-MIMO\codes\bsce\data\raw3gpp_quadriga\';
  quadriga_make_H(q, [o 'H_3gpp_upa'],  'UPA',   100000);
  quadriga_make_H(q, [o 'H_3gpp_ulay'], 'ULA_Y', 100000);
  ```
- **Speed:** about 70 ms per channel in Octave here, so about 2 h per array. `'parts', 1:10` and `'parts', 11:20` in two windows halve it.
- **Fixes:** three fixes of its deviations from 38.901 are on by default: `zsd_fix`, `czsd_fix`, `zod_offset` (section 5.2).
- **Not fixed:** its own placement of the cluster angles. Its horizontal ULA therefore stays less sparse than the TR's (97 against 64 beams).

Then, in the bsce environment:
```
%run -G scripts/make_dataset.py --dataset 3GPP --array UPA --input "data/raw3gpp/H_3gpp_upa_part*.mat" --sizes sv --authors-bands --name 3GPP_UPA_sionna
```
With `--input`, the validation and test sets are the last drops of the files, not drops 1,000,001 / 2,000,001.

---

## 4. Choices worth knowing
- **ULA axis.**
  - `ULA_Y` (horizontal, `3GPP_ULA`) is the usual 3GPP array. Its beams resolve azimuth, as the SV ULA does.
  - `ULA` on z (vertical, `3GPP_ULAZ`) repeats the geometry of GM-LAMP's DeepMIMO ULA. It resolves elevation, where the UMi channel is very concentrated.
  - Report `3GPP_ULA`. Use `3GPP_ULAZ` only for a like-for-like comparison with the DeepMIMO ULA.
- **LOS / NLOS.**
  - The default mixes LOS and NLOS by the standard's LOS probability: 39 % LOS in this cell.
  - `umi_los` / `umi_nlos` isolate them, for example to show where a method gains.
- **Tables.**
  - `--spec 16.1` (default) is the 38.901 version most published work and QuaDRiGa use.
  - Rel-19 (38.901 v19) updated the UMi DS, ASD, ASA and ZSA for 0.5–100 GHz; `--spec 19.2` uses them.
  - At 28 GHz they change the beamspace little (section 5.5).
- **Element.** The 38.901 element is the standard's. `omni` is the setting of SV.
- **What the generator leaves out, and why h does not change.**
  - The UE antenna is isotropic, V-polarised and not rotated, so only the departure side enters h.
  - The arrival angles, their ray coupling and the cross-polarisation ratio are therefore not drawn. They would multiply each ray by a factor of modulus 1 (isotropic) or 0 (the H-polarised part), so h is the same with or without them.
  - Sionna draws them and gives the same statistics (section 5.2).
- **Another scenario.** UMa, for example, is not offered: the generators are written for UMi. The other scenarios need their own LOS probability, heights and ISD (UMa: 25 m, 500 m), and other tables.

---

## 5. The checks of this session
All numbers come from 10,000 channels per array and generator (UPA, ULA on y, ULA on z), made with the default
settings. The test sets are 2,000 users of held-out drops.

The figures are in `docs/3gpp_check/` and print their numbers:

| folder | what it holds |
|---|---|
| `builtin/` | `check_3gpp.py --builtin` on the built-in generator |
| `builtin_v19/` | the same with the Rel-19 tables |
| `sionna/`, `sionna_v19/`, `quadriga/` | the same on the other generators |
| `generators/` | all of them compared |

### 5.1 The standard
`builtin/lsp_vs_38901.png`, `builtin/los_k_vs_38901.png`.

**LOS probability** (Table 7.4.2-1, per 2-D distance bin; 10,000 distinct users):

| | generated | expected | binomial sd |
|---|---|---|---|
| built-in | 0.395 | 0.393 | 0.005 |
| Sionna | 0.395 | 0.393 | 0.005 |
| QuaDRiGa | 0.390 | 0.394 | 0.005 |

Every distance bin lies on the 38.901 curve within ±2 sd. On 100,000 UPA channels the built-in LOS fraction is 0.3937 against 0.3943 expected (z = −0.4).

**Large-scale parameters** (Table 7.5-6 v16.1 at 28 GHz; h_BS 10 m, h_UT 1.5 m). The table gives medians, with the offset of the drawn median from the table in decades:

| state | parameter | 38.901 | built-in | Sionna | QuaDRiGa |
|---|---|---|---|---|---|
| LOS | DS | 32.3 ns | 30.6 ns (−0.024) | 33.2 ns (+0.012) | 32.5 ns (+0.002) |
| LOS | ASD | 13.7° | 13.1° (−0.020) | 14.0° (+0.010) | 13.7° (−0.001) |
| LOS | ASA | 41.0° | 39.8° (−0.013) | 42.4° (+0.015) | 40.6° (−0.004) |
| LOS | ZSD (distance law) | 1.22° | 1.23° (+0.004) | 1.19° (−0.017) | 1.26° (+0.010) |
| LOS | ZSA | 3.83° | 3.80° (−0.004) | 3.78° (−0.007) | 3.86° (+0.002) |
| NLOS | DS | 65.9 ns | 69.7 ns (+0.024) | 64.9 ns (−0.007) | 64.7 ns (−0.008) |
| NLOS | ASD | 15.6° | 15.3° (−0.008) | 16.0° (+0.010) | 16.1° (+0.013) |
| NLOS | ASA | 49.3° | 49.7° (+0.003) | 50.8° (+0.013) | 48.8° (−0.004) |
| NLOS | ZSD (distance law) | 0.87° | 0.84° (−0.020) | 0.90° (+0.012) | 0.87° (−0.004) |
| NLOS | ZSA | 7.27° | 7.31° (+0.002) | 7.48° (+0.012) | 7.33° (+0.003) |
| LOS | K-factor | 9 dB, sd 5 dB | 9.27 dB, 4.82 dB | 8.95 dB, 5.23 dB | 8.96 dB, 5.09 dB |

- **Why 10,000 users give offsets up to ±0.024 decades:** the LSPs are spatially correlated within a drop (correlation distances of 7–13 m in a 133 m cell), so 10,000 users are far fewer independent samples. Sionna's offsets are of the same size.
- **On 100,000 built-in UPA channels** every offset is within ±0.012 decades, the log-sd ratios are 0.99–1.01, and the K-factor is 9.06 dB with sd 4.97 dB.
- **Rel-19 tables** (`builtin_v19/`, `--spec 19.2`): every median is within the same range of the updated values.

### 5.2 Three implementations of the same standard
`generators/generators.png`, `generators/generators.md` (the first 10,000 channels of each):

| array | generator | beams for 95 % (median, p10–p90) | LOS / NLOS median | top-1 energy | LMMSE at 10 dB |
|---|---|---|---|---|---|
| UPA | **built-in** | **16 (7–38)** | 20 / 15 | 0.41 | −11.07 dB |
| UPA | Sionna 2.2 | 17 (7–39) | 19 / 16 | 0.41 | −11.03 dB |
| UPA | QuaDRiGa + fixes | 24 (10–49) | 21 / 27 | 0.38 | −10.50 dB |
| UPA | QuaDRiGa as shipped | 56 (15–77) | 25 / 66 | 0.31 | −9.05 dB |
| ULA y | **built-in** | **65 (7–118)** | 13 / 87 | 0.31 | −3.18 dB |
| ULA y | Sionna 2.2 | 64 (7–119) | 13 / 88 | 0.30 | −3.19 dB |
| ULA y | QuaDRiGa + fixes | 97 (8–144) | 19 / 117 | 0.30 | −3.02 dB |
| ULA y | QuaDRiGa as shipped | 97 (8–142) | 18 / 115 | 0.30 | −3.04 dB |
| ULA z | **built-in** | **8 (3–16)** | 7 / 8 | 0.52 | −11.31 dB |
| ULA z | Sionna 2.2 | 8 (3–17) | 7 / 8 | 0.51 | −11.31 dB |
| ULA z | QuaDRiGa + fixes | 9 (4–18) | 8 / 10 | 0.48 | −10.77 dB |
| ULA z | QuaDRiGa as shipped | 44 (7–54) | 13 / 48 | 0.34 | −9.23 dB |
| UPA, v19.2 tables | built-in | 16 (7–38) | 19 / 15 | 0.41 | −11.15 dB |
| UPA, v19.2 tables | Sionna 2.2 | 17 (8–39) | 19 / 16 | 0.41 | −11.17 dB |

**The built-in generator and Sionna agree.**
- Sparsity: the medians of the beams holding 95 % differ by at most 1 beam, and the p10–p90 ranges by at most 1.
- Top-1 energy: within 0.006.
- LMMSE at 10 dB: within 0.04 dB.
- The two implementations share no code and use different random numbers.

**QuaDRiGa 2.8.1 deviates from 38.901 in four places** (its code and configuration files, read here):
1. **The LOS ZSD mean lacks the term 0.01·|h_UT − h_BS|** of Table 7.5-6 (UMi). That is +0.085 decades for 10 m / 1.5 m. Fixed by `'zsd_fix'`.
2. **The ray zeniths of departure.**
   - 38.901: θ_n,m,ZOD = θ_n,ZOD + (3/8)·10^μlgZSD·α_m (eq. 7.5-20), about 0.2–2° here.
   - QuaDRiGa: a fixed 4° (`PerClusterES_D = 4`), so each cluster spans about ±8.6° in elevation.
   - Fixed by `'czsd_fix'`.
3. **The NLOS ZOD offset of Table 7.5-6** (−10^(−1.5·log10(max(10, d2D)) + 3.3)°) is not applied. Fixed by `'zod_offset'`.
4. **The cluster angles.**
   - QuaDRiGa scales random angles so that each user's realised spread equals its drawn spread.
   - The TR's eq. 7.5-9 to 7.5-19 give smaller realised spreads, so sparser channels in azimuth.
   - Not fixed. With QuaDRiGa's own strict-TR branch switched on in a local copy, its horizontal ULA goes from 99 to 63 beams, against Sionna's 63 (`generators/strict_mapping_test/README.md`).

### 5.3 Compared with SV and DeepMIMO
`builtin/beamspace_3datasets.png`, `builtin/beamspace_upa_maps.png`, `builtin/lmmse_3datasets.png` (test sets of 2,000 channels):

| array | data set | beams for 95 % (median, p10–p90) | top-1 energy | LMMSE NMSE at 0 / 10 / 20 dB |
|---|---|---|---|---|
| UPA | SV | 30 (12–53) | 0.38 | −3.2 / −7.4 / −11.4 dB |
| UPA | DeepMIMO | 15 (4–33) | 0.58 | −9.0 / −16.6 / −25.0 dB |
| UPA | **3GPP** | **17 (7–39)** | 0.41 | −5.4 / −11.1 / −17.1 dB |
| ULA | SV | 9 (4–16) | 0.48 | −1.6 / −3.1 / −3.4 dB |
| ULA | DeepMIMO (z) | 7 (2–11) | 0.63 | −7.6 / −15.0 / −23.5 dB |
| ULA | **3GPP, y (horizontal)** | **65 (7–118)** | 0.30 | −1.5 / −3.2 / −3.7 dB |
| ULA | 3GPP, z (vertical) | 8 (4–16) | 0.51 | −5.7 / −11.3 / −17.2 dB |

- **UPA.**
  - 3GPP is about as sparse as DeepMIMO, and sparser than SV.
  - Almost all of its energy sits in one or two vertical beams. The zenith spread at the BS is about 1°, and the users are 4–40° below the BS (`beamspace_upa_maps.png`).
  - The horizontal beams cover the ±60° sector.
- **Horizontal ULA.**
  - It is far less sparse than any other set: a median of 65 beams, and 118 for the 10 % least sparse channels.
  - With 256 elements, an NLOS cluster of 20 rays (intra-cluster ASD 10°) covers tens of beams.
  - This is the case where a method's prior matters most.
- **Vertical ULA.** It is as sparse as GM-LAMP's DeepMIMO ULA, which has the same geometry.
- **LMMSE** (sample covariance of 6,000 training channels, the same number for every set) measures what the second-order statistics alone give:
  - the 3GPP UPA and vertical ULA sit between SV and DeepMIMO;
  - the horizontal 3GPP ULA is like SV, about −3 dB, because its energy is spread over the whole sector.

### 5.4 The authors' GM-LAMP networks on the 3GPP test sets
`builtin/transfer_gmlamp.png`: the published networks, each with its own CS matrix. The 0–10 dB networks are tested at 5 dB, the 10–20 dB networks at 15 dB (NMSE, dB):

| network | own test set (UPA / ULA) | 3GPP UPA | 3GPP ULA y | 3GPP ULA z |
|---|---|---|---|---|
| SV-trained, 0–10 dB | −12.7 / −13.7 | −3.0 | −2.6 | −8.8 |
| SV-trained, 10–20 dB | −18.6 / −20.8 | −0.7 | −1.1 | −12.6 |
| DeepMIMO-trained, 0–10 dB | −18.1 / −19.0 | +1.6 | −0.4 | −6.1 |
| DeepMIMO-trained, 10–20 dB | −25.8 / −26.6 | +0.6 | 0.0 | −8.7 |

- **A network trained on SV or DeepMIMO loses 10–26 dB on 3GPP data.** It does better than nothing only on the vertical ULA, the 3GPP set that resembles DeepMIMO's.
- **So the 3GPP sets are a real third test, not a repetition** of SV and DeepMIMO. Every method must be trained on them, with the same protocol (3.4).
- The Sionna and QuaDRiGa data give the same picture (`sionna/check_3gpp.md`, `quadriga/check_3gpp.md`).

### 5.5 v16.1 or v19.2 tables
UPA, built-in and Sionna (5,000 channels each with `--spec 19.2`): 16 and 17 beams for 95 %, against 16 and 17 with v16.1. LMMSE at 10 dB: −11.15 and −11.17 dB, against −11.07 and −11.03 dB.

At 28 GHz the Rel-19 update moves the UMi spreads by a few per cent, and the beamspace hardly changes.
- **v16.1 stays the default,** because most published work and QuaDRiGa use it.
- **A v19.2 copy** (`--spec 19.2 --suffix _v19`) answers a reviewer who asks for the latest version.

### 5.6 End to end
- **Data:** `make_3gpp_data.py --check` made `data/3GPP_UPA` and `data/3GPP_ULA` (80,000 / 2,000 / 2,000, LOS 0.392).
- **Training and evaluation:** `train.py` (LADMM, T = 1, a short run) and `evaluate.py` with AMP ran on `3gpp_upa.yaml` with no error.
- **Result:** NMSE −4.8 dB at 0 dB and −9.0 dB at 10 dB, against AMP's −2.9 and −9.5 dB.

### 5.7 What this means for the paper
- **Report `3GPP_UPA` and `3GPP_ULA` next to SV and DeepMIMO,** trained with exactly the same protocols.
  - The UPA is comparable to DeepMIMO in sparsity, but with the standard's statistics.
  - The horizontal ULA tests the methods where the channel is not sparse.
- **State the configuration of section 2,** with "3GPP TR 38.901 v16.1, UMi street canyon, 28 GHz, narrowband".
- **State the generator:** bsce's implementation of the TR procedure, checked against Sionna 2.2.
- **State the sizes** (80,000 / 2,000 / 2,000) and the held-out-drop test set.

---

## 6. Limits
- **One sector of one site.** There is no interference and no wrap-around: this is a channel-estimation data set, not a system-level one.
- **Narrowband.** One frequency, as the SV and DeepMIMO data. A wideband set would need the cluster delays (drawn in step 5, used here only for the cluster powers) and an OFDM grid.
- **Plane waves, as the 38.901 baseline.**
  - A 256-element λ/2 array at 28 GHz is 1.37 m long. Its Rayleigh distance (2D²/λ ≈ 350 m) is larger than the cell, so the users are physically in its near field.
  - SV and DeepMIMO v1.3 are plane-wave too, so the comparison is fair. A near-field data set is a separate study (Rel-19 added optional near-field and spatial non-stationarity models; not used).
- **20 rays per cluster** (the baseline).
  - With 256 elements the array resolves the rays of a cluster, which is part of why these channels are less sparse than 3-path data.
  - Rel-19 allows a variable number of rays per cluster for very large arrays (optional, not used).
- **Random numbers.**
  - With the same NumPy version and seed, the built-in generator gives the same channels on every computer. Another NumPy version may give other channels with the same statistics.
  - Sionna and QuaDRiGa give other channels for the same seed. Their statistics are what is comparable.

---

## 7. Files and tests

| file | what it is |
|---|---|
| `bsce/data/tr38901.py` | the built-in generator (TR 38.901 §7.4–7.5, UMi; v16.1 / v19.2) |
| `scripts/make_3gpp_data.py` | the 3GPP data sets in one command |
| `scripts/make_dataset.py --dataset 3GPP` | one data set: built-in (no `--input`) or from raw files (`--input`) |
| `configs/scenarios/3gpp_upa.yaml`, `3gpp_ula.yaml`, `3gpp_ulaz.yaml` | the scenarios |
| `scripts/check_3gpp.py` | the checks of section 5.1, 5.3, 5.4 (`--builtin` or `--raw`) |
| `scripts/compare_3gpp_generators.py` | the comparison of section 5.2 |
| `scripts/sionna_make_H.py`, `matlab/quadriga_make_H.m` | the second implementations |
| `docs/3gpp_check/` | the figures, `.md` and `.json` of section 5 |
| `tests/test_tr38901_builtin.py` | the built-in generator and its data sets (13 tests) |
| `tests/test_make_dataset_3gpp.py` | raw-file data sets, the table formulas, Sionna and QuaDRiGa when installed |

**What `tests/test_tr38901_builtin.py` checks:**
- **Reproducibility:** drops made alone equal the same drops made in sequence.
- **The array factor:** the factorised array sum equals the direct sum for the three arrays.
- **Geometry:** the users lie in the sector.
- **Statistics:** the LOS count and the LSP medians and spreads agree with the v16.1 and v19.2 tables.
- **The data set:** the lens and normalisation, held-out drops with the same validation and test users for any training size, the meta file, the ULA axis and the size presets.
- **The scripts:** `make_3gpp_data.py` (dry run, refusals, a small real run), `check_3gpp.py --builtin`, `run_suite.py --scenarios all 3gpp` and the shipped scenario files.
