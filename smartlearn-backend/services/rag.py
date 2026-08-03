"""
RAG (Retrieval-Augmented Generation) helpers for SmartLearn.

Reusable pipeline:
    PDF -> pages -> chunks -> embeddings -> saved artifacts

Usage from a notebook:
    from services import rag
    pages = rag.extract_pages_for_rag("Day3/pdf1.pdf")
    chunks = rag.build_chunks(pages, chunk_mode="character_overlap", chunk_size=700, overlap=120)
    bundle = rag.ensure_artifacts("pdf1", "pdf1.pdf", pages, ...)
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

_SOFT_HYPHEN = "­"
_NULL_BYTE = "\x00"


def clean_text(text: str) -> str:
    """Normalise one extracted page of PDF text.

    Removes null bytes, soft hyphens, repeated whitespace, and noisy
    line breaks so downstream chunking sees clean sentences.
    """
    if not text:
        return ""

    # Remove null bytes and soft hyphens
    text = text.replace(_NULL_BYTE, "").replace(_SOFT_HYPHEN, "")

    # Collapse 3+ consecutive newlines to at most 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces / tabs (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove spaces at the start of each line (common PDF artifact)
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)

    # Strip leading/trailing whitespace from the whole text
    text = text.strip()

    # Remove lines that are pure whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text


# ---------------------------------------------------------------------------
# 2. Page loading
# ---------------------------------------------------------------------------


def extract_pages_for_rag(
    file_path: str | Path,
    page_limit: int | None = None,
) -> list[dict]:
    """Read a PDF from disk and return one ``{page, text}`` record per page.

    Parameters
    ----------
    file_path:
        Path to the PDF file.
    page_limit:
        Optional upper bound on the number of pages to read.
        ``None`` means read every page.

    Returns
    -------
    list[dict]
        Each dict has keys ``"page"`` (1-indexed) and ``"text"``.
        Pages whose extracted text is empty after cleaning are skipped.
    """
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        if page_limit is not None and len(records) >= page_limit:
            break
        raw = (page.extract_text() or "").strip()
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


# ---------------------------------------------------------------------------
# 3. Chunking helpers
# ---------------------------------------------------------------------------


def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into smaller pieces.

    Tries natural boundaries first (paragraph, sentence, word) and only
    falls back to a character-level split when no better boundary exists.
    """
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        # Try to split at natural boundaries within the chunk_size window
        split_at = _find_boundary(
            remaining, chunk_size,
            ["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " "],
        )
        if split_at <= 0:
            split_at = chunk_size  # hard fallback — no boundary found
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        pieces.append(remaining)

    return [p for p in pieces if p]


def _find_boundary(text: str, limit: int, separators: list[str]) -> int:
    """Return the best split position ≤ *limit* using the first separator that matches."""
    for sep in separators:
        pos = text.rfind(sep, 0, limit)
        if pos > 0:
            return pos + len(sep)
    return 0


def chunk_by_paragraph(
    records: list[dict],
    chunk_size: int,
    **_kwargs: Any,
) -> list[dict]:
    """Convert page records into chunks that respect paragraph boundaries.

    Each paragraph (text between double-newlines) becomes one chunk.
    A paragraph longer than *chunk_size* is split further via
    :func:`slice_long_text`.
    """
    chunks: list[dict] = []
    for rec in records:
        page_num = rec["page"]
        paragraphs = rec["text"].split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= chunk_size:
                chunks.append({"page": page_num, "text": para})
            else:
                for piece in slice_long_text(para, chunk_size):
                    chunks.append({"page": page_num, "text": piece})
    return chunks


def chunk_by_characters(
    records: list[dict],
    chunk_size: int,
    overlap: int = 0,
    **_kwargs: Any,
) -> list[dict]:
    """Create fixed-size sliding-window chunks.

    Parameters
    ----------
    overlap:
        Number of characters shared between consecutive windows.
        Set to 0 for plain non-overlapping windows.
    """
    stride = max(1, chunk_size - overlap)
    chunks: list[dict] = []
    for rec in records:
        page_num = rec["page"]
        text = rec["text"]
        start = 0
        while start < len(text):
            window = text[start : start + chunk_size].strip()
            if window:
                chunks.append({"page": page_num, "text": window})
            start += stride
    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[dict]:
    """Dispatch to the requested chunking strategy and add chunk ids.

    Parameters
    ----------
    records:
        Page records from :func:`extract_pages_for_rag`.
    chunk_mode:
        One of ``"paragraph"``, ``"character"``, or ``"character_overlap"``.
    chunk_size:
        Target maximum characters per chunk.
    overlap:
        Character overlap between consecutive windows (only meaningful for
        ``"character_overlap"`` mode).

    Returns
    -------
    list[dict]
        Each dict has ``chunk_id``, ``page``, ``text``, and ``chunk_mode``.
    """
    _valid_modes = {"paragraph", "character", "character_overlap"}
    if chunk_mode not in _valid_modes:
        raise ValueError(
            f"chunk_mode must be one of {sorted(_valid_modes)}, got {chunk_mode!r}"
        )

    if chunk_mode == "paragraph":
        raw = chunk_by_paragraph(records, chunk_size=chunk_size)
        effective_overlap = 0
    elif chunk_mode == "character":
        raw = chunk_by_characters(records, chunk_size=chunk_size, overlap=0)
        effective_overlap = 0
    else:  # character_overlap
        raw = chunk_by_characters(records, chunk_size=chunk_size, overlap=overlap)
        effective_overlap = overlap

    # Assign chunk ids and attach metadata
    total = len(raw)
    zfill = max(4, len(str(total)))
    result: list[dict] = []
    for i, item in enumerate(raw, start=1):
        result.append(
            {
                "chunk_id": f"chunk-{i:0{zfill}d}",
                "page": item["page"],
                "text": item["text"],
                "chunk_mode": chunk_mode,
            }
        )
    return result


# ---------------------------------------------------------------------------
# 4. JSON helpers
# ---------------------------------------------------------------------------


def save_json(data: Any, path: str | Path) -> None:
    """Save a Python object to a UTF-8 JSON file, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    """Read a JSON artifact back into Python."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 5. Preview helper
# ---------------------------------------------------------------------------


def preview_records(
    records: list[dict],
    columns: list[str] | None = None,
    rows: int = 5,
) -> Any:
    """Return a small table for notebook inspection.

    Uses pandas if available; otherwise prints a plain-text summary.
    """
    if not records:
        return None

    cols = columns or list(records[0].keys())

    try:
        import pandas as pd
    except ImportError:
        # Plain-text fallback
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        for rec in records[:rows]:
            vals = [str(rec.get(c, ""))[:100] for c in cols]
            lines.append(" | ".join(vals))
        print("\n".join(lines))
        return None

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    usable = [c for c in cols if c in frame.columns]
    return frame[usable].head(rows)


# ---------------------------------------------------------------------------
# 6. Embedding helpers
# ---------------------------------------------------------------------------

# Module-level model cache so load_model does not reload the same model
_model_cache: dict[str, Any] = {}


def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix.

    >>> model_tag("sentence-transformers/all-MiniLM-L6-v2")
    'all_MiniLM_L6_v2'
    """
    base = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_]", "_", base)


def resolve_model_source(model_name: str) -> str | None:
    """Return the path to a local model folder if one exists, else ``None``.

    Checks common project-local cache directories so the notebook and
    backend can avoid fetching the model from Hugging Face on first load.
    """
    local_name = model_tag(model_name)
    candidates: list[Path] = []

    # Project backend cache
    try:
        backend_root = Path(__file__).resolve().parent.parent  # smartlearn-backend
        candidates.append(
            backend_root / "artifacts" / "rag" / "hf_models" / local_name
        )
    except (NameError, OSError):
        pass

    # CWD-relative notebook cache (e.g. Day3/artifacts/...)
    candidates.append(Path("Day3") / "artifacts" / "hf_models" / local_name)

    required = [
        "modules.json",
        "config_sentence_transformers.json",
        "1_Pooling/config.json",
    ]
    for candidate in candidates:
        if candidate.exists() and all(
            (candidate / f).exists() for f in required
        ):
            return str(candidate)

    return None


def get_device() -> str:
    """Return ``"cuda"`` when a usable GPU is present, otherwise ``"cpu"``."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_model(
    model_name: str,
    model_cache_dir: str | Path | None = None,
) -> Any:
    """Load (or reuse) a SentenceTransformer model instance.

    Cached at module level so consecutive calls return the same object.
    """
    cache_key = f"{model_name}__{model_cache_dir or ''}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    device = get_device()
    load_kwargs: dict[str, Any] = {"device": device}

    # Prefer local cache dir if provided and it exists
    if model_cache_dir:
        cache_path = Path(model_cache_dir)
        if cache_path.exists():
            model = SentenceTransformer(str(cache_path), **load_kwargs)
            _model_cache[cache_key] = model
            return model

    # Try to resolve a local model source automatically
    local = resolve_model_source(model_name)
    if local:
        model = SentenceTransformer(local, **load_kwargs)
    else:
        model = SentenceTransformer(model_name, **load_kwargs)

    _model_cache[cache_key] = model
    return model


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_cache_dir: str | Path | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode a list of texts into normalised float32 embedding vectors.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), embedding_dim)``, L2-normalised.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = load_model(model_name, model_cache_dir=model_cache_dir)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Artifact paths
# ---------------------------------------------------------------------------


def artifact_paths_for(
    document_id: str,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    artifact_root: str | Path | None = None,
) -> dict[str, Path]:
    """Determine where each pipeline artifact should be saved.

    Returns a dict with keys:
    ``raw_pages``, ``chunks``, ``embeddings``, ``manifest``, ``root``.
    """
    root = Path(artifact_root) if artifact_root else Path("artifacts")
    tag = model_tag(model_name)

    # All artifacts for one document live under a single subdirectory
    doc_dir = root / document_id

    return {
        "root": root,
        "raw_pages": doc_dir / "pages.json",
        "chunks": doc_dir / f"chunks_{chunk_mode}.json",
        "embeddings": doc_dir / f"embeddings_{chunk_mode}_{tag}.npy",
        "manifest": doc_dir / f"manifest_{chunk_mode}_{tag}.json",
    }


def _manifest_signature(
    document_id: str,
    chunk_mode: str,
    chunk_size: int,
    overlap: int,
    model_name: str,
) -> str:
    """Build a short string that uniquely identifies a pipeline configuration."""
    return f"{document_id}|{chunk_mode}|{chunk_size}|{overlap}|{model_name}"


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build (or reuse) the pages → chunks → embeddings → manifest bundle.

    If a manifest already exists and its configuration signature matches,
    cached artifacts are reloaded instead of recomputed.

    Returns a dict with keys ``manifest``, ``chunks``, ``embeddings``.
    """
    paths = artifact_paths_for(
        document_id=document_id,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        artifact_root=artifact_root,
    )
    signature = _manifest_signature(
        document_id, chunk_mode, chunk_size, overlap, model_name
    )

    # --- reuse cached artifacts when the signature still matches ----------
    manifest_path = paths["manifest"]
    if manifest_path.exists():
        try:
            cached = load_json(manifest_path)
            if cached.get("signature") == signature:
                cached_chunks = load_json(paths["chunks"])
                cached_embeddings = np.load(
                    paths["embeddings"], allow_pickle=False
                )
                return {
                    "manifest": cached,
                    "chunks": cached_chunks,
                    "embeddings": cached_embeddings,
                }
        except (OSError, ValueError, KeyError):
            pass  # corrupt or outdated — rebuild

    # --- build fresh artifacts -------------------------------------------
    # 1. Save raw pages
    save_json(pages, paths["raw_pages"])

    # 2. Build chunks
    chunks = build_chunks(
        pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap
    )
    save_json(chunks, paths["chunks"])

    # 3. Embed
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(
        chunk_texts,
        model_name=model_name,
        batch_size=batch_size,
    )

    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(str(paths["embeddings"]), embeddings)

    # 4. Write manifest
    manifest: dict[str, Any] = {
        "signature": signature,
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 0,
        "device": get_device(),
        "batch_size": batch_size,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(manifest, manifest_path)

    return {
        "manifest": manifest,
        "chunks": chunks,
        "embeddings": embeddings,
    }


# ---------------------------------------------------------------------------
# 8. Misc helpers
# ---------------------------------------------------------------------------


def relative_path_str(path: str | Path, base: str | Path) -> str:
    """Return *path* as a string relative to *base* when possible."""
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        return str(path)
