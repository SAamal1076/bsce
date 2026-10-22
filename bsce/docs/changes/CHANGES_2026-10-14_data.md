# 14 Oct 2026 (3): easy data sets of any size / split / SNR range

## New
**`scripts/make_dataset.py`: one command per data set (SV or DeepMIMO, ULA or UPA).** It writes:
- channels train / val / test, with `--n` + `--split` or exact sizes
- the standard test set (default) or a random test split
- measurement files for any SNR bands (`--snr` edges, 100 SNR chunks, `snr_db` per column)
- GM-LAMP's CS matrix
- `dataset_check.png`, `dataset_info.json`
- `configs/scenarios/<name>.yaml`, ready for train / evaluate

Details:
- **SV**: 200k channels take about 1 min. The standard test set equals `data/SV_<ARR>/..._test_channel.mat` bit for bit, and train/val come from independent random streams.
- **DeepMIMO**: GM-LAMP's beamspace convention. The GM-LAMP test users are found (1991/2000, as before) and removed from train/val. If there are too few users, the script says which rows to generate.

**`matlab/deepmimo_make_H_chunks.m`**: raw DeepMIMO channels for a large set in 300-row files.
- It computes the rows needed itself: 200k users → rows 1000–2120.
- A restart keeps the files already made.

**`tests/test_make_dataset.py`**: sizes, standard test set, negative-SNR band names, per-column SNR in the files, scenario file.

## Changed
- `scripts/import_gmlamp_weights.py` always imports the authors' 0–10 / 10–20 dB networks, whatever the bands of the scenario.
- `matlab/deepmimo_make_H.m` warns about the 54k minimum only for the default rows.
- `RUNBOOK.md`: new section **DATA** with the SV and DeepMIMO steps (MATLAB + Python) and the checks.
- **`reference/gmlamp/` is now delivered** as `bsce_reference_part1–3.zip` (extract next to `bsce_code.zip`). It holds the CS matrices, GM-LAMP's DeepMIMO test channels and the authors' trained weights, which the data and comparison scripts need.

## GM-LAMP data in one command
- `scripts/make_gmlamp_data.py` makes SV_ULA, SV_UPA, DeepMIMO_ULA and DeepMIMO_UPA, the GM-LAMP setting.
  - It calls `generate_data.py` / `deepmimo_convert.py`, then checks the fingerprints of the review data.
  - A full regeneration from scratch is bit-identical (channels and measurement files), in about 1.5 min.
- `DATA_GUIDE.md`: the complete guide (setting, needs, commands, expected output, files, MATLAB step).
- RUNBOOK sections: DATA (A) for GM-LAMP and DATA (B) for any size or range.
