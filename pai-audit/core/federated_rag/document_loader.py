"""
Document loader with chunking.

Supports PDF, TXT, MD, CSV, JSON.  Splits documents into overlapping
chunks suitable for embedding and retrieval.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

from .config import DocumentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """A single chunk of text with metadata."""
    content: str
    source: str          # file path or identifier
    chunk_index: int     # position in the original document
    metadata: dict = field(default_factory=dict)  # arbitrary key-value pairs

    def __post_init__(self) -> None:
        # Ensure content is stripped of excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", self.content.strip())
        object.__setattr__(self, "content", cleaned)


class DocumentLoader:
    """Load files from disk and split into chunks."""

    def __init__(self, config: Optional[DocumentConfig] = None) -> None:
        self._config = config or DocumentConfig()

    def load_file(self, path: str | Path) -> List[Document]:
        """Load a single file and return chunked documents."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        ext = path.suffix.lower()
        if ext not in self._config.supported_extensions:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {self._config.supported_extensions}"
            )

        raw_text = self._read_file(path, ext)
        if not raw_text.strip():
            logger.warning("Empty document: %s", path)
            return []

        return list(self._chunk(raw_text, str(path)))

    def load_directory(self, path: str | Path) -> List[Document]:
        """Recursively load all supported files in a directory."""
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        documents: List[Document] = []
        for ext in self._config.supported_extensions:
            for file_path in sorted(path.rglob(f"*{ext}")):
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                    logger.info("Loaded %d chunks from %s", len(docs), file_path.name)
                except Exception:
                    logger.exception("Failed to load %s", file_path)
        return documents

    # ── Private helpers ──────────────────────────────────

    def _read_file(self, path: Path, ext: str) -> str:
        """Read file content based on extension."""
        if ext == ".pdf":
            return self._read_pdf(path)
        elif ext == ".csv":
            return self._read_csv(path)
        elif ext == ".json":
            return self._read_json(path)
        else:
            return path.read_text(encoding="utf-8")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from PDF using pdfplumber (preferred) or PyMuPDF."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            pass

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            pass

        raise ImportError(
            "PDF reading requires pdfplumber or PyMuPDF. "
            "Install with: pip install pdfplumber  or  pip install PyMuPDF"
        )

    @staticmethod
    def _read_csv(path: Path) -> str:
        """Convert CSV rows to readable text."""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return ""

        # Build natural-language descriptions of each row
        columns = list(rows[0].keys())
        lines = [f"Columns: {', '.join(columns)}"]
        for i, row in enumerate(rows):
            parts = [f"{col}: {row[col]}" for col in columns if row.get(col)]
            lines.append(f"Row {i + 1}: {'; '.join(parts)}")
        return "\n".join(lines)

    @staticmethod
    def _read_json(path: Path) -> str:
        """Convert JSON to readable text, flattening nested structures."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        def flatten(obj, prefix=""):
            """Recursively flatten JSON into key-value text lines."""
            lines = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        lines.extend(flatten(v, key))
                    else:
                        lines.append(f"{key}: {v}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    lines.extend(flatten(item, f"{prefix}[{i}]"))
            else:
                lines.append(f"{prefix}: {obj}")
            return lines

        return "\n".join(flatten(data))

    def _chunk(
        self, text: str, source: str
    ) -> Generator[Document, None, None]:
        """Split text into overlapping chunks at paragraph boundaries."""
        chunk_size = self._config.chunk_size
        overlap = self._config.chunk_overlap

        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )

        # Split on paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge small paragraphs into chunks of ~chunk_size
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                # Handle paragraphs longer than chunk_size
                if len(para) > chunk_size:
                    for sub in self._split_long_paragraph(para, chunk_size):
                        chunks.append(sub)
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)

        # Add overlap context to each chunk
        for i, chunk in enumerate(chunks):
            context_parts = []
            if i > 0:
                prev_tail = chunks[i - 1][-overlap:]
                context_parts.append(prev_tail)
            context_parts.append(chunk)
            if i < len(chunks) - 1:
                next_head = chunks[i + 1][:overlap]
                context_parts.append(next_head)

            merged = "\n... ".join(context_parts)
            yield Document(
                content=merged,
                source=source,
                chunk_index=i,
            )

    @staticmethod
    def _split_long_paragraph(text: str, size: int) -> List[str]:
        """Split a long paragraph on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= size:
                current = f"{current} {sent}" if current else sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks if chunks else [text[:size]]
