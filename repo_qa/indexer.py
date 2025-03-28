"""Walk a git repo, split files by language, embed, and persist a FAISS index."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .config import (
    CACHE_ROOT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
    MAX_FILE_BYTES,
)


EXT_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".swift": Language.SWIFT,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".scala": Language.SCALA,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
    ".sol": Language.SOL,
}

# Always-skip suffixes (binaries, lockfiles, minified bundles).
SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".avi", ".webm",
    ".lock", ".min.js", ".min.css",
)


def _hash_path(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]


def index_dir_for(repo_path: Path) -> Path:
    return CACHE_ROOT / _hash_path(repo_path)


def _list_tracked_files(repo_path: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return [repo_path / line for line in result.stdout.splitlines() if line]


def _is_skippable(path: Path) -> bool:
    if path.name.lower().endswith(SKIP_SUFFIXES):
        return True
    try:
        return path.stat().st_size > MAX_FILE_BYTES
    except OSError:
        return True


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _splitter_for(path: Path) -> RecursiveCharacterTextSplitter:
    lang = EXT_TO_LANGUAGE.get(path.suffix.lower())
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def _split_file(path: Path, repo_path: Path, content: str) -> list[Document]:
    rel = path.relative_to(repo_path).as_posix()
    lang = EXT_TO_LANGUAGE.get(path.suffix.lower())
    chunks = _splitter_for(path).split_text(content)
    docs: list[Document] = []
    cursor = 0
    for chunk in chunks:
        # Best-effort line-number metadata. Splitters don't expose offsets,
        # so we re-find each chunk in the source.
        idx = content.find(chunk, cursor)
        if idx == -1:
            idx = content.find(chunk)
        if idx >= 0:
            line_start = content.count("\n", 0, idx) + 1
            cursor = idx + len(chunk)
        else:
            line_start = 1
        line_end = line_start + chunk.count("\n")
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": rel,
                    "language": lang.value if lang else "text",
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )
    return docs


def _iter_documents(repo_path: Path) -> Iterable[Document]:
    for path in _list_tracked_files(repo_path):
        if not path.is_file() or _is_skippable(path):
            continue
        content = _read_text(path)
        if not content or not content.strip():
            continue
        yield from _split_file(path, repo_path, content)


def get_embeddings(model: str = DEFAULT_EMBED_MODEL) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=model)


def build_index(repo_path: Path, *, embed_model: str = DEFAULT_EMBED_MODEL) -> FAISS:
    repo_path = repo_path.resolve()
    docs = list(_iter_documents(repo_path))
    if not docs:
        raise RuntimeError(
            f"No indexable files found under {repo_path}. "
            "Is it a git repo with tracked files?"
        )
    store = FAISS.from_documents(docs, get_embeddings(embed_model))
    out_dir = index_dir_for(repo_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))
    return store


def load_index(repo_path: Path, *, embed_model: str = DEFAULT_EMBED_MODEL) -> FAISS:
    repo_path = repo_path.resolve()
    out_dir = index_dir_for(repo_path)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"No index for {repo_path}. Run: repo-qa index {repo_path}"
        )
    return FAISS.load_local(
        str(out_dir),
        get_embeddings(embed_model),
        # Safe here: we created the file ourselves under ~/.cache/repo-qa.
        allow_dangerous_deserialization=True,
    )
