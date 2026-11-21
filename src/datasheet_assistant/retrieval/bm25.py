import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25Retriever:
    """Keyword-based retrieval over datasheet text chunks."""

    def __init__(self, chunks_path: str | Path):
        self.chunks_path = Path(chunks_path)

        with self.chunks_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.chunks = data.get("chunks", [])

        if not self.chunks:
            raise ValueError("No chunks found in the input JSON file.")

        self.tokenized_chunks = [
            tokenize(chunk["text"]) for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top matching chunks for a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
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

    def expand_context(self, results: list[dict], window: int = 1) -> list[dict]:
        """Add neighbouring chunks from the same document page around each retrieved chunk.

        Neighbours are the chunks up to `window` positions before and after a retrieved chunk
        in the original `self.chunks` order, restricted to the same (doc_id, page_number).
        Retrieved chunks keep their original fields and BM25 score and get "expanded": False;
        added neighbours get "score": 0.0 and "expanded": True. No chunk_id appears twice.
        Output is grouped by (doc_id, page_number) in order of first appearance in `results`,
        and ordered by original chunk order within each page. The input is not modified.
        """
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}")

        pages = {}  # (doc_id, page_number) -> chunks of that page, in self.chunks order
        for chunk in self.chunks:
            pages.setdefault((chunk["doc_id"], chunk["page_number"]), []).append(chunk)

        retrieved_by_page = {}  # same key -> {position in page: retrieved result}
        for result in results:
            key = (result["doc_id"], result["page_number"])
            siblings = pages.get(key)
            if siblings is None:
                raise ValueError(f"no chunks found for doc_id={key[0]!r}, page_number={key[1]!r}")
            position = next((i for i, c in enumerate(siblings) if c["chunk_id"] == result["chunk_id"]), None)
            if position is None:
                raise ValueError(
                    f"chunk_id {result['chunk_id']!r} is not on doc_id={key[0]!r}, page_number={key[1]!r}"
                )
            retrieved_by_page.setdefault(key, {}).setdefault(position, result)  # first occurrence wins

        expanded = []
        for key, retrieved in retrieved_by_page.items():
            siblings = pages[key]
            wanted = set()
            for position in retrieved:
                wanted.update(range(max(0, position - window), min(len(siblings), position + window + 1)))
            for position in sorted(wanted):
                if position in retrieved:
                    item = dict(retrieved[position])
                    item["expanded"] = False
                else:
                    chunk = siblings[position]
                    item = {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "page_number": chunk["page_number"],
                        "section_heading": chunk.get("section_heading"),
                        "score": 0.0,
                        "text": chunk["text"],
                        "expanded": True,
                    }
                expanded.append(item)
        return expanded