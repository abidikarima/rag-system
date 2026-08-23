from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddedChunk:
    """
    A chunk together with its embedding representation.
    """
    chunk_id: int
    text: str
    embedding: list[float]
    source_chunk: Any
    metadata: dict[str, Any]


class DoclingChunkEmbedder:
    """
    Generate embeddings for Docling chunks.

    Parameters
    ----------
    model_name:
        SentenceTransformer model name/path.

    batch_size:
        Number of chunks encoded at once.

    normalize_embeddings:
        If True, embeddings are L2-normalized.
        This is useful when using cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ):
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        model_kwargs = {}

        if device is not None:
            model_kwargs["device"] = device

        self.model = SentenceTransformer(
            model_name,
            **model_kwargs,
        )

    def _get_text(self, chunk: Any) -> str:
        """
        Extract text from a Docling chunk.
        """

        if hasattr(chunk, "text"):
            return chunk.text

        if isinstance(chunk, str):
            return chunk

        raise TypeError(
            f"Unsupported chunk type: {type(chunk)}"
        )

    def _get_metadata(self, chunk: Any) -> dict[str, Any]:
        """
        Extract useful metadata from a Docling chunk.
        """

        if not hasattr(chunk, "meta"):
            return {}

        meta = chunk.meta

        metadata = {}

        if hasattr(meta, "headings"):
            metadata["headings"] = meta.headings

        if hasattr(meta, "captions"):
            metadata["captions"] = meta.captions

        if hasattr(meta, "doc_items"):
            metadata["doc_items"] = [
                {
                    "ref": getattr(item, "self_ref", None),
                    "label": str(getattr(item, "label", "")),
                }
                for item in meta.doc_items
            ]

        return metadata

    def embed(
        self,
        chunks: Sequence[Any],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        texts = [
            self._get_text(chunk)
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        return [
            EmbeddedChunk(
                chunk_id=i,
                text=texts[i],
                embedding=embeddings[i].tolist(),
                source_chunk=chunks[i],
                metadata=self._get_metadata(chunks[i]),
            )
            for i in range(len(chunks))
        ]

    def embed_vectors(
        self,
        chunks: Sequence[Any],
    ) -> np.ndarray:
        """
        Return only the embedding matrix.

        Shape:
            (number_of_chunks, embedding_dimension)
        """

        texts = [
            self._get_text(chunk)
            for chunk in chunks
        ]

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
