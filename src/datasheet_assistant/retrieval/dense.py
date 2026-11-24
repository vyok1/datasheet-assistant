"""Dense semantic retrieval over datasheet text chunks using sentence-transformers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence
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

    def __init__(
        self,
        chunks_source: str | Path | Sequence[dict[str, Any]],
        embedder: Embedder | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
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

        # Precompute chunk embeddings once during initialization
        texts = [chunk["text"] for chunk in self.chunks]
        raw_embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        self.chunk_embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        if len(self.chunk_embeddings.shape) != 2 or self.chunk_embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"Embedding count ({self.chunk_embeddings.shape[0]}) does not match "
                f"chunk count ({len(self.chunks)})."
            )

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

        if query_vec.ndim > 1:
            query_vec = query_vec[0]

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
