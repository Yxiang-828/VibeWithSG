"""Render REPORT.md to REPORT.pdf using reportlab.

Minimal-dependency (no pandoc/weasyprint): supports headings (#..####),
paragraphs, bullet lists (- item), numbered lists (1. item), fenced code
blocks (``` ... ```), inline code (`x`), bold (**x**), italics (*x*),
simple pipe tables, and ![alt](path) images.

Usage:  python scripts/export_report_pdf.py [MD] [PDF]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether,
                                Preformatted)
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parents[1]
MD = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "REPORT.md"
PDF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "REPORT.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=8, textColor=colors.HexColor("#102040"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1e3a8a"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=6, spaceAfter=4, textColor=colors.HexColor("#1e3a8a"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=4, alignment=TA_LEFT)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontName="Courier", fontSize=8, leading=10, backColor=colors.HexColor("#f2f2f5"), borderPadding=4, spaceAfter=6)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)


def inline(s: str) -> str:
    # images handled upstream; here just format
    s = s.replace("&", "&amp;")
    s = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = s.replace("×", "&#215;").replace("≈", "&#8776;").replace("→", "&#8594;").replace("⇒", "&#8658;")
    s = s.replace("≥", "&#8805;").replace("·", "&middot;").replace("—", "&#8212;").replace("–", "&#8211;")
    return s


def make_table(rows):
    tbl_rows = []
    for r in rows:
        tbl_rows.append([Paragraph(inline(c), BODY) for c in r])
    t = Table(tbl_rows, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4ecf8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#808080")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def parse(md: str):
    """Yield flowables from markdown text."""
    flow = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # fenced code
        if line.strip().startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j]); j += 1
            flow.append(Preformatted("\n".join(buf), CODE))
            flow.append(Spacer(1, 4))
            i = j + 1; continue

        # pipe-table: header then |---|---|
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-\|:]+\|\s*$", lines[i+1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if re.match(r"^[\s\-:]+$", "".join(cells)):
                    i += 1; continue
                rows.append(cells); i += 1
            flow.append(make_table(rows))
            flow.append(Spacer(1, 6))
            continue

        # headings
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            level = len(m.group(1)); text = m.group(2)
            if level == 1: flow.append(Paragraph(inline(text), H1))
            elif level == 2: flow.append(Paragraph(inline(text), H2))
            else: flow.append(Paragraph(inline(text), H3))
            i += 1; continue

        # hr
        if line.strip() == "---":
            flow.append(Spacer(1, 6)); i += 1; continue

        # image ![alt](path)
        m = re.match(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$", line)
        if m:
            path = (ROOT / m.group(1)).resolve()
            if path.exists():
                try:
                    img = Image(str(path))
                    # scale to fit 16cm width
                    max_w = 16 * cm
                    iw, ih = img.imageWidth, img.imageHeight
                    if iw > max_w:
                        scale = max_w / iw
                        img.drawWidth = max_w; img.drawHeight = ih * scale
                    flow.append(img); flow.append(Spacer(1, 6))
                except Exception as e:
                    flow.append(Paragraph(f"[image failed: {path.name}: {e}]", BODY))
            else:
                flow.append(Paragraph(f"[missing image: {m.group(1)}]", BODY))
            i += 1; continue

        # bullets
        if re.match(r"^\s*[-*]\s+", line):
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                t = re.sub(r"^\s*[-*]\s+", "", lines[i])
                flow.append(Paragraph(f"&bull; {inline(t)}", BULLET))
                i += 1
            continue

        # numbered
        if re.match(r"^\s*\d+\.\s+", line):
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                t = re.sub(r"^\s*(\d+\.)\s+", r"\1 ", lines[i])
                flow.append(Paragraph(inline(t), BULLET))
                i += 1
            continue

        # blank
        if not line.strip():
            flow.append(Spacer(1, 3)); i += 1; continue

        # paragraph (possibly multi-line)
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|!\[|\s*[-*]\s|\s*\d+\.\s|---)", lines[i]):
            buf.append(lines[i]); i += 1
        flow.append(Paragraph(inline(" ".join(buf)), BODY))

    return flow


def main():
    md = MD.read_text()
    flow = parse(md)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title="VibeWithSG Segmentation Report")
    doc.build(flow)
    print(f"wrote {PDF} ({PDF.stat().st_size/1024:.1f} KB, {doc.page} pages)")


if __name__ == "__main__":
    main()
