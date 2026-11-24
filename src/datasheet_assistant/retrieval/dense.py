"""Dense semantic retrieval over datasheet text chunks using sentence-transformers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol, Sequence
import zipfile

import numpy as np


class Embedder(Protocol):
    """Protocol for embedding generation (real model or test mock)."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Encode a list of texts into a 2D numpy array of shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    """Wrapper around SentenceTransformer with local CPU execution."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        # Prefer local cache to avoid network probes/retries in offline environments
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class DenseRetriever:
    """Dense semantic retrieval over datasheet text chunks using cosine similarity."""

    CACHE_VERSION = 1

    def __init__(
        self,
        chunks_source: str | Path | Sequence[dict[str, Any]],
        embedder: Embedder | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        embeddings_cache: str | Path | None = None,
    ) -> None:
        if isinstance(chunks_source, (str, Path)):
            self.chunks_path: Path | None = Path(chunks_source)
            with self.chunks_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            self.chunks: list[dict[str, Any]] = data.get("chunks", [])
        else:
            self.chunks_path = None
            self.chunks = list(chunks_source)

        if not self.chunks:
            raise ValueError("No chunks found in the input JSON file.")

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = SentenceTransformerEmbedder(model_name=model_name, device=device)

        cache_identity = {
            "version": self.CACHE_VERSION,
            "model": getattr(self.embedder, "cache_key", None)
            or getattr(self.embedder, "model_name", model_name),
            "embedder": f"{type(self.embedder).__module__}.{type(self.embedder).__qualname__}",
            "device": getattr(self.embedder, "device", device),
        }
        chunk_fingerprint = hashlib.sha256(
            json.dumps(
                [(str(chunk["chunk_id"]), chunk["text"]) for chunk in self.chunks],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_key = hashlib.sha256(
            json.dumps(
                {**cache_identity, "chunks_sha256": chunk_fingerprint},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path = Path(embeddings_cache) if embeddings_cache is not None else None
        if (
            cache_path is not None
            and self.chunks_path is not None
            and cache_path.resolve() == self.chunks_path.resolve()
        ):
            raise ValueError("Embedding cache path must not overwrite the chunks source file.")

        model = getattr(self.embedder, "model", None)
        dimension_getter = getattr(model, "get_sentence_embedding_dimension", None)
        expected_dim = getattr(self.embedder, "embedding_dim", None)
        if expected_dim is None and callable(dimension_getter):
            expected_dim = dimension_getter()

        cached_embeddings = None
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    if (
                        int(cache["cache_version"]) == self.CACHE_VERSION
                        and str(cache["cache_key"].item()) == cache_key
                    ):
                        candidate = np.asarray(cache["embeddings"], dtype=np.float32)
                        if self._valid_embeddings(candidate, expected_dim):
                            cached_embeddings = candidate
            except (EOFError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
                cached_embeddings = None

        if cached_embeddings is not None:
            self.chunk_embeddings = cached_embeddings
        else:
            texts = [chunk["text"] for chunk in self.chunks]
            raw_embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            self.chunk_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
            if not self._valid_embeddings(self.chunk_embeddings, expected_dim):
                raise ValueError(
                    "Embeddings must be a finite 2D array with one row per chunk "
                    "and the model's expected embedding dimension."
                )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_cache(cache_path, cache_key)

    def _valid_embeddings(self, embeddings: np.ndarray, expected_dim: int | None) -> bool:
        return (
            embeddings.ndim == 2
            and embeddings.shape[0] == len(self.chunks)
            and embeddings.shape[1] > 0
            and (expected_dim is None or embeddings.shape[1] == expected_dim)
            and bool(np.isfinite(embeddings).all())
        )

    def _write_cache(self, cache_path: Path, cache_key: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as cache_file:
                np.savez_compressed(
                    cache_file,
                    cache_version=np.asarray(self.CACHE_VERSION),
                    cache_key=np.asarray(cache_key),
                    embeddings=self.chunk_embeddings,
                )
            Path(temporary_name).replace(cache_path)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of chunk embeddings."""
        return int(self.chunk_embeddings.shape[1]) if self.chunk_embeddings.ndim == 2 else 0

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top matching chunks for a query using cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        # Encode query with normalization
        query_vec = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True),
            dtype=np.float32,
        )

        if query_vec.ndim == 2 and query_vec.shape[0] == 1:
            query_vec = query_vec[0]
        if query_vec.ndim != 1 or query_vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding must have shape ({self.embedding_dim},), "
                f"got {query_vec.shape}."
            )

        # Dot product of L2-normalized vectors is cosine similarity
        scores = np.dot(self.chunk_embeddings, query_vec)

        # Deterministic ranking: sort by (-score, index)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        results = []
        for index in ranked_indices:
            chunk = self.chunks[index]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "page_number": chunk["page_number"],
                    "section_heading": chunk.get("section_heading"),
                    "score": float(scores[index]),
                    "text": chunk["text"],
                }
            )

        return results
