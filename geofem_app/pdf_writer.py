"""Small dependency-free PDF text writer shared by report modules."""

from __future__ import annotations

from pathlib import Path


def write_text_pdf(path: str | Path, lines: list[str], *, title: str) -> None:
    """Write a minimal UTF-16 text PDF without external dependencies."""

    page_size = (595, 842)
    lines_per_page = 48
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[""]]
    objects: list[tuple[int, bytes]] = []
    page_ids: list[int] = []
    objects.extend(
        [
            (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
            (
                3,
                b"<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiKakuGo-W5 /Encoding /UniJIS-UCS2-H /DescendantFonts [4 0 R] >>",
            ),
            (
                4,
                b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /HeiseiKakuGo-W5 /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 6 >> /FontDescriptor 5 0 R >>",
            ),
            (
                5,
                b"<< /Type /FontDescriptor /FontName /HeiseiKakuGo-W5 /Flags 4 /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 700 /StemV 80 >>",
            ),
        ]
    )
    next_id = 6
    for page_index, page_lines in enumerate(pages, start=1):
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content = _pdf_page_content(page_lines, page_index, len(pages), title)
        objects.append(
            (
                page_id,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_size[0]} {page_size[1]}] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii"),
            )
        )
        objects.append((content_id, b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids).encode("ascii")
    objects.insert(1, (2, b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode("ascii") + b" >>"))
    objects.sort(key=lambda item: item[0])
    max_id = max(object_id for object_id, _body in objects)
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {0: 0}
    for object_id, body in objects:
        offsets[object_id] = len(pdf)
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for object_id in range(1, max_id + 1):
        pdf.extend(f"{offsets.get(object_id, 0):010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    Path(path).write_bytes(bytes(pdf))


def _pdf_page_content(lines: list[str], page_index: int, page_count: int, title: str) -> bytes:
    commands = ["q", "BT", "/F1 10 Tf", "42 800 Td", "14 TL"]
    for line in lines:
        commands.append(f"<{_pdf_text_hex(line)}> Tj")
        commands.append("T*")
    commands.extend(["ET", "BT", "/F1 8 Tf", "42 28 Td", f"<{_pdf_text_hex(f'{title}  p.{page_index}/{page_count}')}> Tj", "ET", "Q"])
    return "\n".join(commands).encode("ascii")


def _pdf_text_hex(text: str) -> str:
    return (b"\xfe\xff" + str(text).encode("utf-16-be", errors="replace")).hex().upper()


__all__ = ["write_text_pdf"]
