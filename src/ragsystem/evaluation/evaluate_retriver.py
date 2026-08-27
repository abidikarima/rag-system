from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragsystem.core.retriever import ChromaRetriever


TEST_DATA_DIR = Path("./test_data/clean_test_data.jsonl")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

TOP_K = 5

# Number of candidates retrieved by each retriever
BM25_TOP_K = 5
VECTOR_TOP_K = 5

# Hybrid ranking weights
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5


def load_test_data(
    test_file: Path,
) -> list[dict[str, Any]]:
    """
    Load test examples from JSONL.
    """

    if not test_file.is_file():
        raise FileNotFoundError(
            f"Test data file not found: {test_file}"
        )

    with test_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
        ]


def normalize_source(
    source: str,
) -> str:
    """
    Normalize source paths before comparison.

    This avoids problems such as:

        data/file.pdf
        ./data/file.pdf
    """

    return str(
        Path(source)
    ).replace("\\", "/")


def source_matches(
    retrieved_source: str | None,
    expected_source: str,
) -> bool:
    """
    Check whether retrieved source matches expected source.
    """

    if not retrieved_source:
        return False

    return (
        normalize_source(
            retrieved_source
        )
        == normalize_source(
            expected_source
        )
    )


def evaluate(
    retriever: ChromaRetriever,
    examples: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """
    Evaluate hybrid BM25 + vector retrieval.

    A question is considered a hit if the expected source
    appears anywhere in the final top-k results.
    """

    total = len(examples)

    if total == 0:
        raise ValueError(
            "No test examples found."
        )

    top_k_hits = 0

    results = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["question"]

        expected_source = (
            example["metadata"]["source"]
        )

        question_id = example.get(
            "question_id",
            example.get(
                "id",
                str(index),
            ),
        )

        # =====================================================
        # HYBRID SEARCH
        # =====================================================

        retrieved = retriever.hybrid_search(
            query=question,
            top_k=top_k,
            bm25_top_k=BM25_TOP_K,
            vector_top_k=VECTOR_TOP_K,
        )

        # =====================================================
        # Check expected source
        # =====================================================

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

        # =====================================================
        # Save detailed result
        # =====================================================

        retrieved_results = []

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):

            retrieved_results.append(
                {
                    "rank": rank,
                    "id": result.id,
                    "source": result.metadata.get(
                        "source"
                    ),
                    "distance": result.distance,
                    "bm25_score": result.bm25_score,
                    "vector_score": result.vector_score,
                    "hybrid_score": result.hybrid_score,
                    "retrieval_sources": (
                        result.retrieval_sources
                    ),
                    "headings": result.metadata.get(
                        "headings"
                    ),
                    "chunk_index": result.metadata.get(
                        "chunk_index"
                    ),
                }
            )

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "hit": hit,
                "retrieved_results": retrieved_results,
            }
        )

        # =====================================================
        # Console output
        # =====================================================

        status = "✓" if hit else "✗"

        print()
        print(
            f"[{status}] {index}/{total} "
            f"{question_id}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: {expected_source}"
        )

        print()
        print("Retrieved:")

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):

            source = result.metadata.get(
                "source",
                "UNKNOWN",
            )

            bm25_score = result.bm25_score
            vector_score = result.vector_score
            hybrid_score = result.hybrid_score

            retrieval_sources = (
                result.retrieval_sources
                or []
            )

            print(
                f"  {rank}. {source}"
            )

            print(
                f"      hybrid={_format_score(hybrid_score)} "
                f"bm25={_format_score(bm25_score)} "
                f"vector={_format_score(vector_score)}"
            )

            print(
                f"      retrieved_by={retrieval_sources}"
            )

            print(
                f"      distance={result.distance}"
            )

    # =========================================================
    # Accuracy
    # =========================================================

    accuracy = (
        top_k_hits / total
    )

    return {
        "total": total,
        "hits": top_k_hits,
        "misses": total - top_k_hits,
        "accuracy": accuracy,
        "top_k": top_k,
        "bm25_top_k": BM25_TOP_K,
        "vector_top_k": VECTOR_TOP_K,
        "bm25_weight": BM25_WEIGHT,
        "vector_weight": VECTOR_WEIGHT,
        "results": results,
    }


def _format_score(
    score: float | None,
) -> str:
    """
    Format scores for readable console output.
    """

    if score is None:
        return "None"

    return f"{score:.4f}"


def main():

    print("=" * 80)
    print("HYBRID RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    print()
    print("Configuration:")
    print(f"  Final TOP_K:     {TOP_K}")
    print(f"  BM25 TOP_K:      {BM25_TOP_K}")
    print(f"  Vector TOP_K:    {VECTOR_TOP_K}")
    print(f"  BM25 weight:     {BM25_WEIGHT}")
    print(f"  Vector weight:   {VECTOR_WEIGHT}")

    # =========================================================
    # Load test data
    # =========================================================

    examples = load_test_data(
        TEST_DATA_DIR
    )

    print()
    print(
        f"Loaded {len(examples)} test questions."
    )

    # =========================================================
    # Initialize retriever
    # =========================================================

    retriever = ChromaRetriever(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,

        # Final number returned
        top_k=TOP_K,

        # BM25 candidate pool
        bm25_top_k=BM25_TOP_K,

        # Vector candidate pool
        vector_top_k=VECTOR_TOP_K,

        # Hybrid weights
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
    )

    # =========================================================
    # Evaluate
    # =========================================================

    evaluation = evaluate(
        retriever=retriever,
        examples=examples,
        top_k=TOP_K,
    )

    # =========================================================
    # Print summary
    # =========================================================

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Questions: {evaluation['total']}"
    )

    print(
        f"Hits:     {evaluation['hits']}"
    )

    print(
        f"Misses:   {evaluation['misses']}"
    )

    print(
        f"Accuracy@{TOP_K}: "
        f"{evaluation['accuracy']:.2%}"
    )

    print("=" * 80)

    # =========================================================
    # Save detailed results
    # =========================================================

    output_path = Path(
        "hybrid_retrieval_evaluation.json"
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

    print()
    print(
        f"Detailed results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
