from __future__ import annotations

import json
import uuid
from pathlib import Path

import chromadb
from ragsystem.core.rag_pipeline.vectorizer import DoclingChunkEmbedder
from ragsystem.core.rag_pipeline.chunker import DoclingChunker, ChunkingOptions
from ragsystem.core.rag_pipeline.injestor import RepositoryDocumentParser


DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"


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


def main():
    # ---------------------------------------------------------
    # 1. Initialize Chroma
    # ---------------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    try:
        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )

    # ---------------------------------------------------------
    # 2. Initialize ingestor
    # ---------------------------------------------------------

    parser = RepositoryDocumentParser(
        data_dir=DATA_DIR,
        enable_ocr=False,
        enable_tables=True,
        accurate_tables=False,
    )

    # ---------------------------------------------------------
    # 3. Initialize chunker
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
    # 4. Initialize embedding model
    # ---------------------------------------------------------

    embedder = DoclingChunkEmbedder(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        batch_size=32,
        normalize_embeddings=True,
    )

    # 5. Process every source file

    for file_path in parser.files():

        print()
        print("=" * 80)
        print(f"Processing: {file_path}")
        print("=" * 80)

        try:
            document = parser.parse_file(file_path)

            # Set metadata source for this document
            chunker.options.source = str(file_path)

            chunks = chunker.chunks(document)

            for chunk in chunks:
                print("-" * 80)
                print(f"Chunk: {chunk.text}...")

                metadata = chunker.metadata(
                    chunk,
                    document=document,
                )

                print(f"Metadata: {metadata}")

            if not chunks:
                print("No chunks generated.")
                continue

            print(f"Chunks: {len(chunks)}")

            embedded_chunks = embedder.embed(chunks)

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
                    embedded.metadata,
                    source_file=str(file_path),
                    chunk_index=index,
                )

                metadata["is_table"] = str(
                    DoclingChunker.is_table(
                        embedded.source_chunk
                    )
                )

                ids.append(chunk_id)
                documents.append(embedded.text)
                embeddings.append(embedded.embedding)
                metadatas.append(metadata)

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
            print(f"Failed: {file_path}")
            print(f"Error: {exc}")

    # ---------------------------------------------------------
    # 6. Show final collection size
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("Pipeline complete")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents in Chroma: {collection.count()}")
    print(f"Chroma path: {Path(CHROMA_DIR).resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
