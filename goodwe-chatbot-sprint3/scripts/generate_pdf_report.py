"""Converte docs/relatorio_evolucao.md em PDF estilizado.

Gera docs/relatorio_evolucao.pdf com reportlab (Platypus), parseando um
subconjunto de Markdown: títulos, parágrafos, listas, negrito e tabelas.

Uso:
    python scripts/generate_pdf_report.py
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = PROJECT_ROOT / "docs" / "relatorio_evolucao.md"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "relatorio_evolucao.pdf"

PAGE_MARGIN = 1.8 * cm
MAX_CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN

TEXT_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"
MONO_FONT = "Courier"

STYLES = {
    "title": ParagraphStyle(
        "title",
        fontName=BOLD_FONT,
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=colors.HexColor("#0B3C5D"),
    ),
    "meta": ParagraphStyle(
        "meta",
        fontName=TEXT_FONT,
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=4,
        textColor=colors.HexColor("#333333"),
    ),
    "h1": ParagraphStyle(
        "h1",
        fontName=BOLD_FONT,
        fontSize=12.5,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#0B3C5D"),
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName=BOLD_FONT,
        fontSize=10.5,
        leading=14,
        spaceBefore=7,
        spaceAfter=4,
        textColor=colors.HexColor("#12557E"),
    ),
    "body": ParagraphStyle(
        "body",
        fontName=TEXT_FONT,
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    ),
    "bullet": ParagraphStyle(
        "bullet",
        fontName=TEXT_FONT,
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        leftIndent=14,
        bulletIndent=2,
        spaceAfter=3,
        bulletFontName=TEXT_FONT,
    ),
    "subbullet": ParagraphStyle(
        "subbullet",
        fontName=TEXT_FONT,
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        leftIndent=28,
        bulletIndent=16,
        spaceAfter=2,
        bulletFontName=TEXT_FONT,
    ),
}

TABLE_HEADER_BG = colors.HexColor("#0B3C5D")
TABLE_ALT_BG = colors.HexColor("#F2F6FA")
TABLE_GRID = colors.HexColor("#B9C4CE")


def _rich(text: str) -> str:
    """Converte **negrito** e `code` do Markdown para markup do reportlab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("```", "&#96;&#96;&#96;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`+([^`]+)`+", lambda m: f"<font face='{MONO_FONT}'>{m.group(1)}</font>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def _split_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def _is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|[\s:\-|]+\|\s*$", line)) and "-" in line


def _merge_soft_wraps(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        if (
            re.match(r"^\s{2,}\S", line)
            and merged
            and merged[-1]
            and not stripped.startswith("|")
            and not merged[-1].startswith("|")
            and not stripped.startswith("#")
        ):
            merged[-1] += " " + stripped
        else:
            merged.append(line)
    return merged


def _build_elements(lines: list[str]) -> list:
    elements = []
    lines = _merge_soft_wraps(lines)
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped == "---":
            elements.append(Spacer(1, 4))
            index += 1
            continue

        if stripped.startswith("# "):
            elements.append(Paragraph(_rich(stripped[2:]), STYLES["title"]))
            index += 1
            continue

        if stripped.startswith("## "):
            elements.append(Paragraph(_rich(stripped[3:]), STYLES["h1"]))
            index += 1
            continue

        if stripped.startswith("### "):
            elements.append(Paragraph(_rich(stripped[4:]), STYLES["h2"]))
            index += 1
            continue

        if stripped.startswith("|"):
            rows = []
            while index < total and lines[index].strip().startswith("|"):
                row = _split_table_row(lines[index])
                if not _is_table_separator(lines[index]):
                    rows.append([_rich(cell) for cell in row])
                index += 1

            if rows:
                header = rows[0]
                body = rows[1:]
                col_count = max(len(header), max((len(r) for r in body), default=0))
                header = header + [""] * (col_count - len(header))
                body = [row + [""] * (col_count - len(row)) for row in body]

                col_widths = _fit_column_widths(header, body)

                cell_style = ParagraphStyle(
                    "cell",
                    fontName=TEXT_FONT,
                    fontSize=8.5,
                    leading=11,
                    alignment=TA_JUSTIFY,
                    spaceAfter=0,
                )
                header_style = ParagraphStyle(
                    "cell_header",
                    fontName=BOLD_FONT,
                    fontSize=8.5,
                    leading=11,
                    alignment=TA_LEFT,
                    textColor=colors.white,
                    spaceAfter=0,
                )
                table_data = [
                    [Paragraph(cell, header_style) for cell in header],
                    *[
                        [Paragraph(cell, cell_style) for cell in row]
                        for row in body
                    ],
                ]

                table = Table(
                    table_data,
                    colWidths=col_widths,
                    hAlign="LEFT",
                    repeatRows=1,
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.4, TABLE_GRID),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_BG]),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                elements.append(Spacer(1, 2))
                elements.append(table)
                elements.append(Spacer(1, 8))
            continue

        if re.match(r"^\s{3,}-\s+", line):
            elements.append(
                Paragraph(
                    _rich(stripped[2:]) if stripped.startswith("- ") else _rich(stripped.lstrip("- ")),
                    STYLES["subbullet"],
                    bulletText="-",
                )
            )
            index += 1
            continue

        if stripped.startswith("- "):
            elements.append(
                Paragraph(
                    _rich(stripped[2:]),
                    STYLES["bullet"],
                    bulletText="-",
                )
            )
            index += 1
            continue

        elements.append(Paragraph(_rich(stripped), STYLES["body"]))
        index += 1

    return elements


def _fit_column_widths(header: list[str], body: list[list[str]]) -> list[float]:
    width_pt = {}
    for col_index in range(len(header)):
        header_chars = len(header[col_index].replace("<b>", "").replace("</b>", ""))
        max_chars = max(
            [header_chars]
            + [
                len(
                    row[col_index]
                    .replace("<b>", "")
                    .replace("</b>", "")
                    if col_index < len(row)
                    else 0
                )
                for row in body
            ]
        )
        width_pt[col_index] = min(max(max_chars * 5.2, 40), 300)

    total = sum(width_pt.values())
    if total > MAX_CONTENT_WIDTH:
        scale = MAX_CONTENT_WIDTH / total
        width_pt = {k: v * scale for k, v in width_pt.items()}
    return [width_pt[i] for i in range(len(header))]


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(TEXT_FONT, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


def main() -> None:
    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    elements = _build_elements(lines)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        title="Relatório de Evolução do Projeto — EV Challenge GoodWe (Sprint 3)",
        author="FIAP — Turma 2SEM 2026.2",
        subject="Prompt and Artificial Intelligence",
    )
    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)
    print(f"PDF gerado: {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()