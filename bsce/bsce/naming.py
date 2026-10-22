"""Run-folder names, one scheme for run_suite.py, train.py and finetune_bands.py (v24).

Trained network (one folder per run, in outputs/<SCENARIO>[_white]/):
    <METHOD>-T<T>[_<protocol>][_tr<edges>][_<option>...][_s<seed>]
    LADMM-T6                      LADMM, 6 layers, one network for 0-20 dB, the default protocol (ladmm_best)
    LAMP-T8_paper_trscenario      LAMP, 8 layers, its own protocol, the scenario's bands (0-10, 10-20 dB)
    NF-LADMM-SR-T4_tr0-10-20      two networks, 0-10 and 10-20 dB
Fine-tuned per band (a folder NEXT TO its source, in the same outputs/<SCENARIO>/):
    <source folder>_ft_<band edges>dB_<iterations>[_<option>...]
    LADMM-T6_ft_0-10-20dB_20k     LADMM-T6 fine-tuned on 0-10 and 10-20 dB, 20,000 iterations per band
    LADMM-T6_ft_0-5-10-15-20dB_20k_margin1
  the files inside: <folder>_<band>.pt, e.g. LADMM-T6_ft_0-10-20dB_20k_0to10dB.pt
Up to v23 the fine-tuned folder was <source>-ft[<edges>] (LADMM-T6-ft, LADMM-T6-ft0-5-10-15-20); such folders are still
found (legacy_ft_tags) and can be renamed with scripts/rename_runs.py.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

DEFAULT_PROTOCOL = "ladmm_best"


def fmt(v: float) -> str:
    """A number for a name: 10 -> '10', -5 -> 'm5', 2.5 -> '2p5'."""
    return f"{float(v):g}".replace("-", "m").replace(".", "p")


def edges_str(edges: Sequence[float]) -> str:
    """[0, 10, 20] -> '0-10-20' (negative edges as m5)."""
    return "-".join(fmt(e) for e in edges)


def band_names(edges: Sequence[float]) -> List[str]:
    """[0, 10, 20] -> ['0to10dB', '10to20dB'] (the band names of the network files)."""
    return [f"{fmt(a)}to{fmt(b)}dB" for a, b in zip(list(edges)[:-1], list(edges)[1:])]


def iters_str(n: int) -> str:
    """20000 -> '20k', 15000 -> '15k', 1500 -> '1500it'."""
    n = int(n)
    return f"{n // 1000}k" if n % 1000 == 0 and n >= 1000 else f"{n}it"


def option_suffix(setting: str) -> str:
    """A --set override as a name part: 'train.restore_best=true' -> 'restore-best-true' (every run of characters other
    than letters, digits and '.' becomes '-'; 'train.lr=0.0005' -> 'lr-0.0005')."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", setting.split(".", 1)[-1]).strip("-")


def run_tag(label: str, T: int, protocol: str = DEFAULT_PROTOCOL, train_edges: Optional[Sequence[float]] = (0, 20),
            seed: int = 0, overrides: Iterable[str] = (), extra: Iterable[str] = ()) -> str:
    """Folder of a trained network. train_edges None = the scenario's bands ('_trscenario');
    extra: suffixes of named options (e2e30k, clip1, margin2, ...), added after the protocol and range."""
    tag = f"{label}-T{int(T)}"
    if protocol and protocol != DEFAULT_PROTOCOL:
        tag += f"_{protocol}"
    if train_edges is None:
        tag += "_trscenario"
    elif [float(e) for e in train_edges] != [0.0, 20.0]:
        tag += f"_tr{edges_str(train_edges)}"
    for e in extra:
        tag += f"_{e}"
    for s in overrides:
        tag += "_" + option_suffix(s)
    if seed:
        tag += f"_s{int(seed)}"
    return tag


def data_extra(margin: float = 0.0, val_snr=None, snr_chunks: Optional[int] = None) -> List[str]:
    """The name parts of the data options of run_suite.py / train.py: _margin2, _valrange / _val10, _c10."""
    out = []
    if margin:
        out.append(f"margin{fmt(margin)}")
    if val_snr:
        vs = [str(v).lower() for v in val_snr]
        out.append("valrange" if vs == ["range"] else "val" + "-".join(fmt(float(v)) for v in val_snr))
    if snr_chunks:
        out.append(f"c{int(snr_chunks)}")
    return out


def ft_tag(source: str, edges: Sequence[float] = (0, 10, 20), iters: int = 20000, margin: float = 0.0,
           extra: Iterable[str] = ()) -> str:
    """Folder of a fine-tune of `source` (a run-folder name) per band."""
    tag = f"{source}_ft_{edges_str(edges)}dB_{iters_str(iters)}"
    if margin:
        tag += f"_margin{fmt(margin)}"
    for e in extra:
        tag += f"_{e}"
    return tag


def legacy_ft_tags(source: str, edges: Sequence[float] = (0, 10, 20), margin: float = 0.0) -> List[str]:
    """The names a v23-or-older run_suite gave the same fine-tune (the budget was not in the name)."""
    t = source + "-ft" + ("" if [float(e) for e in edges] == [0.0, 10.0, 20.0] else edges_str(edges))
    return [t + (f"_margin{fmt(margin)}" if margin else "")]


FT_RE = re.compile(r"^(?P<src>.+)_ft_(?P<edges>[0-9mp-]+)dB_(?P<iters>\d+k|\d+it)(?P<rest>(_.*)?)$")
LEGACY_FT_RE = re.compile(r"^(?P<src>.+)-ft(?P<edges>[0-9mp]+(-[0-9mp]+)+)?(?P<rest>(_.*)?)$")


def parse_ft(name: str):
    """'LADMM-T6_ft_0-10-20dB_20k' -> dict(src, edges, iters, rest, legacy=False); legacy '-ft' names too; else None."""
    m = FT_RE.match(name)
    if m:
        it = m["iters"]
        n = int(it[:-1]) * 1000 if it.endswith("k") else int(it[:-2])
        return dict(src=m["src"], edges=_edges(m["edges"]), iters=n, rest=m["rest"], legacy=False)
    m = LEGACY_FT_RE.match(name)
    if m:
        return dict(src=m["src"], edges=_edges(m["edges"]) if m["edges"] else [0.0, 10.0, 20.0], iters=None,
                    rest=m["rest"], legacy=True)
    return None


def _edges(s: str) -> List[float]:
    return [float(p.replace("m", "-").replace("p", ".")) for p in s.split("-")]


def drop_seed(name: str) -> str:
    """The name without its training seed: LADMM-T6_s1 -> LADMM-T6, LADMM-T6_s2_ft_0-10-20dB_20k -> LADMM-T6_ft_...,
    LADMM-T6_s1-ft (legacy) -> LADMM-T6-ft (mean +- std over seeds groups these)."""
    return re.sub(r"_s\d+(?=(_ft_.*|-ft[A-Za-z0-9.-]*(_.*)?)?$)", "", name)
