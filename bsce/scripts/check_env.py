#!/usr/bin/env python3
"""Step 0 of MANUAL.md: is this machine ready? (Python packages, GPU, reference files, data sets)

  python scripts/check_env.py
Prints OK / MISSING per item and what to do; nothing is changed.
"""
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def line(ok, what, hint=""):
    print(f"  [{'OK' if ok else '--'}] {what}" + (f"   -> {hint}" if (hint and not ok) else ""))
    return ok


def main():
    print(f"bsce folder : {ROOT}")
    v = ROOT / "VERSION.txt"
    print(f"bsce version: {v.read_text(encoding='utf-8').strip() if v.exists() else '(before 21: no VERSION.txt)'}")
    print(f"python      : {sys.version.split()[0]} ({platform.system()})")
    bad = 0
    print("\npackages")
    for mod, hint in (("numpy", "pip install -r requirements.txt"), ("scipy", "pip install -r requirements.txt"),
                      ("yaml", "pip install pyyaml"), ("matplotlib", "pip install -r requirements.txt"),
                      ("torch", "install PyTorch with CUDA (MANUAL.md section 1)"), ("pytest", "pip install pytest")):
        try:
            m = importlib.import_module(mod)
            line(True, f"{mod} {getattr(m, '__version__', '')}")
        except ImportError:
            bad += not line(False, mod, hint)
    try:
        import torch
        cuda = torch.cuda.is_available()
        line(cuda, f"CUDA GPU: {torch.cuda.get_device_name(0) if cuda else 'not visible'}",
             "training still works on the CPU (slow); for the GPU install the CUDA build of PyTorch")
    except ImportError:
        pass
    print("\nreference files (bsce_reference_part*.zip)")
    for f in ("reference/gmlamp/CE_SV/CSmatrix256128.mat", "reference/gmlamp/CE_DeepMIMO/CSmatrix_128256.mat",
              "reference/gmlamp/CE_DeepMIMO/ULA_test_channel.mat", "reference/gmlamp/CE_DeepMIMO/UPA_test_channel.mat",
              "reference/gmlamp/CE_SV/Trained_GM_LAMP_for_SV_UPA_2561280to10dB.mat",
              "reference/gmlamp/CE_DeepMIMO/Trained_GM_LAMP_for_DeepMIMO_UPA_2561280to10dB.mat"):
        bad += not line((ROOT / f).exists(), f, "extract bsce_reference_part1..4.zip where bsce_code.zip was extracted")
    print("\ndata sets (MANUAL.md section 3)")
    for s in ("SV_ULA", "SV_UPA", "DeepMIMO_ULA", "DeepMIMO_UPA"):
        line((ROOT / "data" / s / f"{s}_training_128_measurements_0to10dB.mat").exists(), f"data/{s}",
             "python scripts/make_gmlamp_data.py --sv --deepmimo-ula <H_ula.mat> --deepmimo-upa <H_upa.mat>")
    for s in ("3GPP_UPA", "3GPP_ULA"):                    # optional third data set
        line((ROOT / "data" / s / f"{s}_training_128_measurements_0to10dB.mat").exists(), f"data/{s} (optional)",
             "python scripts/make_3gpp_data.py")
    print("\n" + ("ready." if not bad else f"{bad} required item(s) missing (see the hints)."))


if __name__ == "__main__":
    main()
