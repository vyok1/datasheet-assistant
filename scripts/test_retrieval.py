
from pathlib import Path

from datasheet_assistant.retrieval.bm25 import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "pages" / "chunks.json"

QUESTIONS = [
    "Which registers configure an STM32 GPIO port?",
    "What types of DMA transfers are supported?",
    "What is the function of MISO and MOSI in SPI?",
    "How does the SPI master select a slave device?",
    "Which signals and registers appear in the USART block diagram?",
]

def main():
    retriever = BM25Retriever(CHUNKS_PATH)

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("-" * 80)

        results = retriever.search(question, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. {result['chunk_id']} "
                f"| page={result['page_number']} "
                f"| score={result['score']:.4f}"
            )
            print(f"Section: {result['section_heading']}")
            print(result["text"][:350].replace("\n", " "))


if __name__ == "__main__":
    main()
