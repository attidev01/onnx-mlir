#!/usr/bin/env python3
"""Convert the Gemmini engineering report markdown to a styled PDF via weasyprint."""

import markdown2
from weasyprint import HTML, CSS
from pathlib import Path

MD = Path("SiliconeSignal_Gemmini_Engineering_Report.md")
OUT = Path("SiliconeSignal_Gemmini_Engineering_Report.pdf")

md_text = MD.read_text(encoding="utf-8")

html_body = markdown2.markdown(
    md_text,
    extras=[
        "tables", "fenced-code-blocks", "header-ids",
        "break-on-newline", "code-friendly", "cuddled-lists",
    ],
)

CSS_STYLE = """
@page {
    size: A4;
    margin: 2.2cm 2.5cm 2.2cm 2.5cm;
    @top-left   { content: "SiliconeSignal Technologies"; font-size: 8pt; color: #1A3A5C; }
    @top-right  { content: "Gemmini Backend Report v0.1.5"; font-size: 8pt; color: #1A3A5C; }
    @bottom-center { content: counter(page); font-size: 9pt; color: #555; }
}
* { box-sizing: border-box; }
body {
    font-family: "DejaVu Serif", Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #222;
    background: #fff;
}
/* ── Cover / Title block ─────────────────────────────────────────────── */
h1:first-of-type {
    font-size: 22pt;
    color: #1A3A5C;
    border-bottom: 3px solid #C8A84B;
    padding-bottom: 8pt;
    margin-top: 0;
}
/* ── Headings ────────────────────────────────────────────────────────── */
h1 { font-size: 18pt; color: #1A3A5C; margin-top: 28pt; margin-bottom: 6pt; border-bottom: 1.5px solid #C8A84B; padding-bottom: 4pt; }
h2 { font-size: 14pt; color: #1A3A5C; margin-top: 18pt; margin-bottom: 5pt; }
h3 { font-size: 11.5pt; color: #2C5282; margin-top: 12pt; margin-bottom: 4pt; }
h4 { font-size: 10.5pt; color: #2C5282; margin-top: 10pt; margin-bottom: 3pt; }
/* ── Tables ──────────────────────────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin: 10pt 0 14pt 0;
    page-break-inside: avoid;
}
th {
    background: #1A3A5C;
    color: #fff;
    padding: 5pt 7pt;
    text-align: left;
    font-weight: bold;
}
td {
    padding: 4pt 7pt;
    border-bottom: 0.5pt solid #C5D5E8;
    vertical-align: top;
}
tr:nth-child(even) td { background: #EAF2FB; }
/* ── Code blocks ─────────────────────────────────────────────────────── */
pre, code {
    font-family: "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 8.2pt;
}
pre {
    background: #F5F5F5;
    border: 0.8pt solid #CCCCCC;
    border-left: 3pt solid #1A3A5C;
    padding: 7pt 10pt;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 8pt 0;
    page-break-inside: avoid;
}
code { background: #F0F4F8; padding: 1pt 3pt; border-radius: 2pt; }
pre code { background: none; padding: 0; }
/* ── Lists ───────────────────────────────────────────────────────────── */
ul, ol { margin: 5pt 0 8pt 0; padding-left: 20pt; }
li { margin-bottom: 3pt; }
/* ── Horizontal rules ────────────────────────────────────────────────── */
hr { border: none; border-top: 1.5pt solid #C8A84B; margin: 18pt 0; }
/* ── Blockquotes (used for callout boxes) ────────────────────────────── */
blockquote {
    border-left: 4pt solid #C8A84B;
    background: #FFF8E7;
    margin: 10pt 0;
    padding: 8pt 12pt;
    font-style: italic;
}
/* ── Strong / emphasis ───────────────────────────────────────────────── */
strong { color: #1A3A5C; }
/* ── Page breaks ─────────────────────────────────────────────────────── */
h1 { page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SiliconeSignal Gemmini Accelerator Backend — Engineering Report</title>
</head>
<body>
{html_body}
</body>
</html>"""

HTML(string=full_html, base_url=".").write_pdf(
    str(OUT),
    stylesheets=[CSS(string=CSS_STYLE)],
)

print(f"PDF written: {OUT}  ({OUT.stat().st_size // 1024} KB)")
