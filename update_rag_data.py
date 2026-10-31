from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import chromadb
from ragsystem.core.rag_pipeline.vectorizer import DoclingChunkEmbedder
from ragsystem.core.rag_pipeline.chunker import DoclingChunker, ChunkingOptions
from ragsystem.core.rag_pipeline.injestor import RepositoryDocumentParser
from ragsystem.utils.data_downloader import DEFAULT_CATEGORIES, ensure_data


DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

DENSE_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # embedding vectors
SPARSE_MODEL_NAME = "naver/splade-cocondenser-selfdistil"  # for sparse vectors

SPARSE_EMBEDDING_KEY = "sparse_embedding"
SUMMARY_SPARSE_EMBEDDING_KEY = "summary_sparse_embedding"

# Maximum number of new documents processed per run.
MAX_DOCUMENTS = 1


def make_chroma_metadata(
    metadata: dict,
    source_file: str,
    chunk_index: int,
) -> dict:
    """
    Convert chunk metadata into Chroma-compatible metadata.
    """

    result = {
        "source": source_file,
        "chunk_index": chunk_index,
    }

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            # Chroma does not accept nested dict/list metadata.
            result[key] = json.dumps(
                value,
                ensure_ascii=False,
            )

    return result


def indexed_sources(collection) -> set[str]:
    """
    Return the source files that already have chunks in Chroma.
    """

    data = collection.get(include=["metadatas"])

    return {
        metadata["source"]
        for metadata in data.get("metadatas") or []
        if metadata and metadata.get("source")
    }


def embed_summaries(
    sparse_embedder,
    summaries: list[str | None],
) -> list[dict | None]:
    """
    SPLADE-encode chunk summaries.

    Empty or missing summaries are not encoded and return None.
    """

    positions = [
        index
        for index, summary in enumerate(summaries)
        if summary and summary.strip()
    ]

    vectors = sparse_embedder.embed_documents(
        [summaries[index] for index in positions]
    )

    result: list[dict | None] = [None] * len(summaries)

    for index, vector in zip(positions, vectors):
        result[index] = vector

    return result


def log_chunk_sparse(
    index: int,
    text_vector: dict | None,
    summary: str | None,
    summary_vector: dict | None,
) -> None:
    """
    Print one line showing which sparse vectors were produced.
    """

    def describe(vector: dict | None) -> str:
        if vector is None:
            return "NOT CONVERTED"

        return f"OK ({len(vector['indices'])} terms)"

    if not (summary and summary.strip()):
        summary_status = "SKIPPED (no summary)"
    else:
        summary_status = describe(summary_vector)

    preview = (summary or "").strip().replace("\n", " ")[:70]

    print(
        f"  chunk {index:>3} | text sparse: {describe(text_vector)}"
        f" | summary sparse: {summary_status}"
        f" | summary: {preview!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the corpus and index it into Chroma."
    )

    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--chroma-dir", default=CHROMA_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)

    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="OHR-Bench categories to download.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download data, only index what is in --data-dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download even if --data-dir already contains PDFs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_DOCUMENTS,
        help="Maximum number of new documents to process (0 = no limit).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the collection and re-index every file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Make sure the data is available locally
    # ---------------------------------------------------------

    if not args.skip_download:
        ensure_data(
            data_dir=args.data_dir,
            categories=args.categories,
            force=args.force_download,
        )

    # ---------------------------------------------------------
    # 2. Initialize Chroma
    # ---------------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path=args.chroma_dir
    )

    if args.rebuild:
        try:
            chroma_client.delete_collection(
                name=args.collection
            )
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={
            "hnsw:space": "cosine",
        },
    )

    already_indexed = indexed_sources(collection)

    # ---------------------------------------------------------
    # 3. Initialize ingestor
    # ---------------------------------------------------------

    parser = RepositoryDocumentParser(
        data_dir=args.data_dir,
        enable_ocr=False,
        enable_tables=True,
        accurate_tables=False,
    )

    files = [
        path
        for path in parser.files()
        if str(path) not in already_indexed
    ]

    remaining = len(files)

    if args.limit > 0:
        files = files[:args.limit]

    print(
        f"{len(already_indexed)} files already indexed, "
        f"{remaining} remaining, processing {len(files)}."
    )

    if not files:
        print("Nothing to do.")
        return

    # ---------------------------------------------------------
    # 4. Initialize chunker
    # ---------------------------------------------------------

    chunker = DoclingChunker(
        ChunkingOptions(
            chunker="hybrid",
            pages=None,
            max_tokens=2000,
            merge_peers=True,
            repeat_table_header=True,
        )
    )

    # ---------------------------------------------------------
    # 5. Initialize embedding model
    # ---------------------------------------------------------

    embedder = DoclingChunkEmbedder(
        model_name=DENSE_MODEL_NAME,
        batch_size=32,
        normalize_embeddings=True,
        sparse_model_name=SPARSE_MODEL_NAME,
    )

    # ---------------------------------------------------------
    # 6. Process every new source file
    # ---------------------------------------------------------

    failed = []

    for position, file_path in enumerate(files, start=1):

        print()
        print("=" * 80)
        print(f"[{position}/{len(files)}] Processing: {file_path}")
        print("=" * 80)

        try:
            document = parser.parse_file(file_path)

            # Set metadata source for this document
            chunker.options.source = str(file_path)

            chunks = chunker.chunks(document)

            if not chunks:
                print("No chunks generated.")
                continue

            print(f"Chunks: {len(chunks)}")

            # Title, pages, captions and LLM summary/keywords.
            chunk_metadatas = [
                chunker.metadata(
                    chunk,
                    document=document,
                )
                for chunk in chunks
            ]

            # Dense + text sparse vectors.
            embedded_chunks = embedder.embed(chunks)

            # Sparse vectors of the chunk summaries.
            summaries = [
                chunk_metadata.get("summary")
                for chunk_metadata in chunk_metadatas
            ]

            summary_vectors = embed_summaries(
                embedder.sparse_embedder,
                summaries,
            )

            print("Sparse conversion:")

            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for index, embedded in enumerate(embedded_chunks):

                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{file_path}:{index}",
                    )
                )

                metadata = make_chroma_metadata(
                    {
                        **embedded.metadata,
                        **chunk_metadatas[index],
                    },
                    source_file=str(file_path),
                    chunk_index=index,
                )

                metadata["is_table"] = str(
                    DoclingChunker.is_table(
                        embedded.source_chunk
                    )
                )

                if embedded.sparse_embedding is not None:
                    metadata[SPARSE_EMBEDDING_KEY] = json.dumps(
                        embedded.sparse_embedding
                    )

                if summary_vectors[index] is not None:
                    metadata[SUMMARY_SPARSE_EMBEDDING_KEY] = json.dumps(
                        summary_vectors[index]
                    )

                log_chunk_sparse(
                    index,
                    embedded.sparse_embedding,
                    summaries[index],
                    summary_vectors[index],
                )

                ids.append(chunk_id)
                documents.append(embedded.text)
                embeddings.append(embedded.embedding)
                metadatas.append(metadata)

            text_converted = sum(
                embedded.sparse_embedding is not None
                for embedded in embedded_chunks
            )

            summary_converted = sum(
                vector is not None
                for vector in summary_vectors
            )

            print(
                f"Sparse vectors: text {text_converted}/{len(embedded_chunks)}, "
                f"summary {summary_converted}/{len(summary_vectors)} chunks converted."
            )

            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            print(
                f"Stored {len(embedded_chunks)} chunks "
                f"in Chroma."
            )

        except Exception as exc:
            failed.append(file_path)
            print(f"Failed: {file_path}")
            print(f"Error: {exc}")

    # ---------------------------------------------------------
    # 7. Show final collection size
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("Pipeline complete")
    print(f"Collection: {args.collection}")
    print(f"Documents in Chroma: {collection.count()}")
    print(f"Chroma path: {Path(args.chroma_dir).resolve()}")

    if failed:
        print(f"Failed files ({len(failed)}):")
        for path in failed:
            print(f"  - {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
