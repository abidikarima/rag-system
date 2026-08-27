from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragsystem.core.retriever import ChromaRetriever


TEST_DATA_DIR = Path("./test_data/test_data.jsonl")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

TOP_K = 5


def load_test_data(test_file: Path) -> list[dict[str, Any]]:
    # Ensure the path is a file
    if not test_file.is_file():
        raise FileNotFoundError(f"Test data file not found: {test_file}")

    # Read the JSONL file
    with test_file.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize_source(source: str) -> str:
    """
    Normalize source paths before comparison.

    This avoids problems such as:
        data/file.pdf
        ./data/file.pdf
    """

    return str(Path(source)).replace("\\", "/")


def source_matches(
    retrieved_source: str | None,
    expected_source: str,
) -> bool:
    if not retrieved_source:
        return False

    return (
        normalize_source(retrieved_source)
        == normalize_source(expected_source)
    )


def evaluate(
    retriever: ChromaRetriever,
    examples: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:

    total = len(examples)

    if total == 0:
        raise ValueError("No test examples found.")

    top_k_hits = 0

    results = []

    for index, example in enumerate(examples, start=1):

        question = example["question"]
        expected_source = example["source"]

        question_id = example.get(
            "question_id",
            example.get("id", str(index)),
        )

        retrieved = retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        retrieved_sources = [
            result.metadata.get("source")
            for result in retrieved
        ]

        hit = any(
            source_matches(
                source,
                expected_source,
            )
            for source in retrieved_sources
        )

        if hit:
            top_k_hits += 1

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            }
        )

        status = "✓" if hit else "✗"

        print()
        print(
            f"[{status}] {index}/{total} "
            f"{question_id}"
        )

        print(f"Question: {question}")
        print(f"Expected: {expected_source}")

        print("Retrieved:")

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):
            source = result.metadata.get(
                "source",
                "UNKNOWN",
            )

            print(
                f"  {rank}. "
                f"{source} "
                f"(distance={result.distance})"
            )

    accuracy = top_k_hits / total

    return {
        "total": total,
        "hits": top_k_hits,
        "misses": total - top_k_hits,
        "accuracy": accuracy,
        "top_k": top_k,
        "results": results,
    }


def main():

    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    examples = load_test_data(
        TEST_DATA_DIR
    )

    print(
        f"Loaded {len(examples)} test questions."
    )

    # ---------------------------------------------------------
    # Initialize retriever
    # ---------------------------------------------------------

    retriever = ChromaRetriever(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    evaluation = evaluate(
        retriever=retriever,
        examples=examples,
        top_k=TOP_K,
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Questions: {evaluation['total']}")
    print(f"Hits:     {evaluation['hits']}")
    print(f"Misses:   {evaluation['misses']}")

    print(
        f"Accuracy@{TOP_K}: "
        f"{evaluation['accuracy']:.2%}"
    )

    print("=" * 80)

    # ---------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------

    output_path = Path(
        "retrieval_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evaluation,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Detailed results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
