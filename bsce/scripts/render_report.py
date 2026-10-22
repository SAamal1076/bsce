#!/usr/bin/env python3
"""Render VERIFICATION_REPORT.md to a self-contained HTML file (figures embedded).

  python scripts/render_report.py   -> VERIFICATION_REPORT.html
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;
     color:#1d1d1b;background:#fff;line-height:1.5;font-size:15px}
h1{font-size:26px;border-bottom:2px solid #2a78d6;padding-bottom:6px} h2{font-size:20px;margin-top:32px;color:#1a4f8f}
table{border-collapse:collapse;margin:12px 0;font-size:13px;display:block;overflow-x:auto}
th,td{border:1px solid #d0d0cc;padding:4px 8px;text-align:left;vertical-align:top} th{background:#eef3fa}
code{background:#f3f3f1;padding:1px 4px;border-radius:3px;font-size:12.5px}
pre{background:#f6f6f4;padding:10px;border-radius:4px;overflow-x:auto} pre code{background:none;padding:0}
img{max-width:100%;border:1px solid #e3e3e0;margin:8px 0}
"""


def main():
    md = (ROOT / "VERIFICATION_REPORT.md").read_text(encoding="utf-8")

    def embed(m):
        alt, path = m.group(1), m.group(2)
        data = base64.b64encode((ROOT / path).read_bytes()).decode()
        return f"![{alt}](data:image/png;base64,{data})"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>bsce verification report</title><style>{CSS}</style></head><body>{body}</body></html>")
    (ROOT / "VERIFICATION_REPORT.html").write_text(html, encoding="utf-8")
    print("wrote", ROOT / "VERIFICATION_REPORT.html")


if __name__ == "__main__":
    main()
