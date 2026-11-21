
import json
from pathlib import Path

from datasheet_assistant.retrieval.bm25 import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]
gold_path = ROOT / "data" / "golden_set.json"
chunks_path = ROOT / "data" / "pages" / "chunks.json"

gold = json.loads(gold_path.read_text(encoding="utf-8"))
retriever = BM25Retriever(chunks_path)

inspect_ids = {
    "q005", "q007", "q011", "q014", "q015",
    "q016", "q017", "q018", "q019", "q020",
}

for question in gold["questions"]:
    if question["id"] not in inspect_ids:
        continue

    print(f"\n{'=' * 80}")
    print(f"{question['id']}: {question['question']}")
    print(f"Expected relevant chunks: {question['relevant_chunk_ids']}")

    results = retriever.search(question["question"], top_k=5)

    for rank, result in enumerate(results, start=1):
        print(
            f"  {rank}. {result['chunk_id']} "
            f"| score={result['score']:.4f} "
            f"| {result['text'][:140].replace(chr(10), ' ')}"
        )
