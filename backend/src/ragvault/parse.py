"""Turn uploaded files into plain text and split the text into overlapping chunks."""

from __future__ import annotations

import io
import re
from pathlib import Path

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}; use PDF, MD or TXT")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n\n".join(pages).strip()
    return data.decode("utf-8", errors="replace").strip()


def guess_title(filename: str, text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if 3 <= len(line) <= 120:
            return line
    return Path(filename).stem


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to `size` chars, carry `overlap` chars over."""
    text = re.sub(r"\r\n?", "\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Split very long paragraphs on sentence boundaries, then hard-wrap if still too long.
    pieces: list[str] = []
    for p in paras:
        if len(p) <= size:
            pieces.append(p)
            continue
        buf = ""
        for s in re.split(r"(?<=[.!?])\s+", p):
            while len(s) > size:
                pieces.append(s[:size])
                s = s[size - overlap :]
            if len(buf) + len(s) + 1 > size and buf:
                pieces.append(buf)
                buf = buf[-overlap:] + " " + s if overlap else s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > size:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            buf = f"{buf}\n\n{piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks
