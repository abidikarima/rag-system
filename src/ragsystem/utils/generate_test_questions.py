"""
Generate retrieval test questions from the chunks stored in Chroma.

- Documents are picked at random, each document at most once.
- In each picked document, one random text chunk is selected.
- An LLM (Groq) writes one short question about that chunk, with a
  difficulty level: easy, medium or hard (balanced across the dataset).
- Every question is appended to the output file as soon as it is created,
  so an interrupted run can be resumed by running the command again.

Run from the repository root:

    uv run python -m ragsystem.utils.generate_test_questions

See src/ragsystem/docs/test_data.md for the full description.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb

from ragsystem.prompts.question_generator import (
    COMPLEXITY_RULES,
    QUESTION_GENERATOR,
)
from ragsystem.utils.llm_connector import GroqLLM


CHROMA_DIR = "./chroma_db_v1"
COLLECTION_NAME = "documents"
OUTPUT_FILE = Path("./test_data/test_questions.jsonl")

# Maximum number of questions (= number of distinct documents used).
MAX_QUESTIONS = 300

# Fixed seed: the same collection gives the same selection.
SEED = 42

COMPLEXITIES = ("easy", "medium", "hard")

# Chunk selection: only text chunks of a reasonable size.
CHUNK_TYPES = ("text",)
MIN_CHUNK_WORDS = 60
MAX_CHUNK_WORDS = 600

# Question constraints.
MAX_QUESTION_WORDS = 25
LLM_ATTEMPTS = 3
LLM_RETRY_DELAY_SECONDS = 10

# Questions must not point at the passage itself.
FORBIDDEN_PATTERNS = re.compile(
    r"\b(passage|excerpt|this text|the text|the document|this document|"
    r"this article|the article|according to|mentioned|chunk)\b",
    re.IGNORECASE,
)

# Metadata copied into the test file (sparse vectors etc. are left out).
METADATA_KEYS = ("source", "chunk_index", "title", "pages", "headings", "type")


# =============================================================
# Chunk selection
# =============================================================

def load_candidates(
    chroma_dir: str,
    collection_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return eligible chunks grouped by source document.
    """

    collection = chromadb.PersistentClient(path=chroma_dir).get_collection(
        name=collection_name
    )

    data = collection.get(include=["documents", "metadatas"])

    by_source: dict[str, list[dict[str, Any]]] = {}

    for chunk_id, text, metadata in zip(
        data["ids"],
        data["documents"],
        data["metadatas"],
    ):
        metadata = metadata or {}
        source = metadata.get("source")
        words = len((text or "").split())

        if (
            not source
            or metadata.get("type", "text") not in CHUNK_TYPES
            or not MIN_CHUNK_WORDS <= words <= MAX_CHUNK_WORDS
        ):
            continue

        by_source.setdefault(source, []).append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": {
                    key: metadata[key]
                    for key in METADATA_KEYS
                    if key in metadata
                },
            }
        )

    print(
        f"Collection: {collection.count()} chunks | "
        f"documents with an eligible chunk: {len(by_source)}"
    )

    return by_source


def pick_chunks(
    by_source: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    One random chunk per document, documents in random order.
    """

    sources = sorted(by_source)
    rng.shuffle(sources)

    return [rng.choice(by_source[source]) for source in sources]


def complexity_plan(
    total: int,
    existing: Counter,
    rng: random.Random,
) -> list[str]:
    """
    Difficulty levels still to generate, balanced and shuffled.
    """

    target = Counter(
        {
            level: total // len(COMPLEXITIES)
            + (1 if index < total % len(COMPLEXITIES) else 0)
            for index, level in enumerate(COMPLEXITIES)
        }
    )

    remaining = [
        level
        for level in COMPLEXITIES
        for _ in range(max(0, target[level] - existing[level]))
    ]

    rng.shuffle(remaining)

    return remaining


# =============================================================
# LLM
# =============================================================

def parse_json(response: str) -> dict[str, Any] | None:
    """
    Extract the first JSON object from the LLM response.
    """

    match = re.search(r"\{.*\}", response or "", re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate(result: dict[str, Any] | None) -> str | None:
    """
    Return an error message, or None if the question is acceptable.
    """

    if not result:
        return "invalid JSON"

    question = str(result.get("question", "")).strip()
    answer = str(result.get("answer", "")).strip()

    if not question or not answer:
        return "missing question or answer"

    if not question.endswith("?"):
        return "question does not end with '?'"

    if len(question.split()) > MAX_QUESTION_WORDS:
        return f"question longer than {MAX_QUESTION_WORDS} words"

    if FORBIDDEN_PATTERNS.search(question):
        return "question refers to the passage"

    return None


def format_headings(metadata: dict[str, Any]) -> str:
    """
    Headings are stored as a JSON list string in Chroma.
    """

    try:
        headings = json.loads(metadata.get("headings") or "[]")
    except json.JSONDecodeError:
        headings = [metadata["headings"]]

    return " > ".join(str(heading) for heading in headings) or "none"


def generate_question(
    llm: GroqLLM,
    chunk: dict[str, Any],
    complexity: str,
) -> dict[str, str] | None:
    """
    Ask the LLM for one question; retry when the output is rejected.
    """

    prompt = QUESTION_GENERATOR.format(
        COMPLEXITY=complexity.upper(),
        COMPLEXITY_RULES=COMPLEXITY_RULES[complexity],
        MAX_WORDS=MAX_QUESTION_WORDS,
        HEADINGS=format_headings(chunk["metadata"]),
        PASSAGE=chunk["text"],
    )

    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            response = llm.generate(
                prompt,
                temperature=0.5,
                max_tokens=2048,
            )
        except RuntimeError as exc:
            # All Groq models failed (often a rate limit): wait and retry.
            print(f"    [attempt {attempt}] LLM error: {exc}")
            time.sleep(LLM_RETRY_DELAY_SECONDS)
            continue

        result = parse_json(response)
        error = validate(result)

        if error is None:
            return {
                "question": result["question"].strip(),
                "answer": str(result["answer"]).strip(),
            }

        print(f"    [attempt {attempt}] rejected: {error}")

    return None


# =============================================================
# Main
# =============================================================

def load_existing(output: Path) -> list[dict[str, Any]]:
    if not output.is_file():
        return []

    with output.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrieval test questions from Chroma chunks."
    )

    parser.add_argument("--chroma-dir", default=CHROMA_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output file and start from zero.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    existing = load_existing(args.output)

    if len(existing) >= args.max_questions:
        print(f"{args.output} already has {len(existing)} questions. Nothing to do.")
        return

    rng = random.Random(args.seed)

    chunks = pick_chunks(
        load_candidates(args.chroma_dir, args.collection),
        rng,
    )

    used_sources = {record["metadata"]["source"] for record in existing}
    chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["source"] not in used_sources
    ]

    plan = complexity_plan(
        args.max_questions,
        Counter(record["complexity"] for record in existing),
        rng,
    )

    print(
        f"Existing questions: {len(existing)} | to generate: {len(plan)} | "
        f"unused documents available: {len(chunks)}"
    )

    if len(chunks) < len(plan):
        print(
            "[WARNING] Not enough distinct documents: "
            f"at most {len(chunks)} new questions can be generated."
        )

    llm = GroqLLM()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    created = len(existing)
    skipped = 0
    chunk_iter = iter(chunks)

    with args.output.open("a", encoding="utf-8") as f:
        for complexity in plan:

            # Take the next unused document; if the LLM fails on it,
            # move on to another document with the same difficulty.
            for chunk in chunk_iter:
                result = generate_question(llm, chunk, complexity)

                if result is None:
                    skipped += 1
                    print(f"    skipped {chunk['metadata']['source']}")
                    continue

                record = {
                    "id": chunk["id"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "complexity": complexity,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                created += 1

                print(
                    f"[{created}/{args.max_questions}] {complexity:<6} | "
                    f"{chunk['metadata']['source']} | {result['question']}"
                )
                break
            else:
                print("No more documents available.")
                break

    final = load_existing(args.output)
    levels = Counter(record["complexity"] for record in final)

    print()
    print("=" * 80)
    print(f"Questions in {args.output}: {len(final)}")
    print(f"By difficulty: {dict(levels)}")
    print(f"Distinct documents: {len({r['metadata']['source'] for r in final})}")
    print(f"Documents skipped (LLM failed or output rejected): {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
