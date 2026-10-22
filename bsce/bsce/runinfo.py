"""Human-readable README.txt in every output folder: what the network is, how it was trained, on which data."""
import datetime
import hashlib
from pathlib import Path


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def write_readme(folder, title, sections):
    """sections: list of (heading, dict or list of lines)."""
    lines = [title, "=" * len(title), f"written {datetime.datetime.now():%Y-%m-%d %H:%M}", ""]
    for head, body in sections:
        lines += [head, "-" * len(head)]
        if isinstance(body, dict):
            w = max([len(str(k)) for k in body] + [1])
            lines += [f"  {str(k).ljust(w)} : {_fmt(v)}" for k, v in body.items()]
        else:
            lines += [f"  {x}" for x in body]
        lines.append("")
    Path(folder, "README.txt").write_text("\n".join(lines), encoding="utf-8")


def digest(path, n=12):
    """Short SHA-1 of a file: two computers used the same data file when the digests are equal."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]
