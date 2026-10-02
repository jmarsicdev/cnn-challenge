"""Render docs/report_draft.md to a compact HTML file suitable for upload to Google Drive
(converted to a Google Doc) or printing to a two-page PDF. Figures are embedded as data URIs.

    python scripts/build_report.py --md docs/report_draft.md --out docs/report.html
"""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import markdown

CSS = """
body{font-family:Arial,Helvetica,sans-serif;font-size:10pt;line-height:1.25;color:#111;max-width:7.1in;margin:0.5in auto}
h1{font-size:15pt;margin:0 0 2pt 0}h2{font-size:11.5pt;margin:9pt 0 3pt 0;border-bottom:1px solid #999;padding-bottom:1pt}
p{margin:0 0 5pt 0}em{color:#444}table{border-collapse:collapse;font-size:8.5pt;margin:4pt 0 6pt 0;width:100%}
th,td{border:1px solid #bbb;padding:1.5pt 4pt;text-align:left;vertical-align:top}th{background:#eee}
figure{margin:4pt 0;text-align:center}figcaption{font-size:8pt;color:#444}img{max-width:100%}
.figrow{display:flex;gap:8pt;align-items:flex-start}.figrow figure{flex:1}
"""


def embed(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="docs/report_draft.md")
    ap.add_argument("--out", default="docs/report.html")
    ap.add_argument("--fig", nargs="*", default=["docs/final/test_confusion.png", "docs/final/lowlabel_curve.png"])
    args = ap.parse_args()
    md = Path(args.md).read_text()
    html = markdown.markdown(md, extensions=["tables"])
    figs = [Path(f) for f in args.fig if Path(f).exists()]
    if figs:
        caps = {"test_confusion.png": "Test confusion matrix of the final model (400 images). Errors cluster in natural scenes.",
                "lowlabel_curve.png": "Low-label study: supervised only vs EMA teacher-student, 3 seeds each."}
        row = '<div class="figrow">' + "".join(
            f'<figure><img src="{embed(f)}"><figcaption>{caps.get(f.name, f.name)}</figcaption></figure>' for f in figs) + "</div>"
        # place figures before the failure-analysis heading
        html = re.sub(r"(<h2>4\. )", row + r"\1", html, count=1)
    Path(args.out).write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>CNN Challenge report</title>"
                              f"<style>{CSS}</style></head><body>{html}</body></html>")
    words = len(re.sub(r"<[^>]+>", " ", html).split())
    print(f"wrote {args.out}: {words} words, {len(figs)} figures")


if __name__ == "__main__":
    main()
