from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer, SparseEncoder


# A sparse vector: vocabulary token ids and their weights.
#     {"indices": [1011, 1999, ...], "values": [0.51, 1.63, ...]}
SparseVectorDict = dict[str, list]


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
    sparse_embedding: SparseVectorDict | None = None


class SpladeEmbedder:
    """
    Generate SPLADE sparse vectors.

    Each text becomes a weighted bag of vocabulary terms
    (about 30k dimensions, only a few hundred non-zero).

    Parameters
    ----------
    model_name:
        SparseEncoder model name/path.

    batch_size:
        Number of texts encoded at once.

    decimals:
        Weights are rounded to keep stored vectors small.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-selfdistil",
        batch_size: int = 32,
        decimals: int = 4,
        device: str | None = None,
    ):
        self.batch_size = batch_size
        self.decimals = decimals

        model_kwargs = {}

        if device is not None:
            model_kwargs["device"] = device

        self.model = SparseEncoder(
            model_name,
            **model_kwargs,
        )

    def _to_dicts(self, embeddings) -> list[SparseVectorDict]:
        """
        Convert a batch of torch sparse tensors to index/value dicts.
        """

        vectors = []

        for row in embeddings:
            row = row.coalesce()

            vectors.append(
                {
                    # Coalesced indices are sorted ascending.
                    "indices": row.indices()[0].cpu().tolist(),
                    "values": [
                        round(value, self.decimals)
                        for value in row.values().cpu().tolist()
                    ],
                }
            )

        return vectors

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseVectorDict]:

        if not texts:
            return []

        embeddings = self.model.encode_document(
            list(texts),
            batch_size=self.batch_size,
            convert_to_sparse_tensor=True,
            show_progress_bar=True,
        )

        return self._to_dicts(embeddings)

    def embed_query(
        self,
        query: str,
    ) -> SparseVectorDict:

        embeddings = self.model.encode_query(
            [query],
            convert_to_sparse_tensor=True,
        )

        return self._to_dicts(embeddings)[0]

    def decode(
        self,
        vector: SparseVectorDict,
        top_k: int | None = 10,
    ) -> list[tuple[str, float]]:
        """
        Return the highest-weighted tokens of a sparse vector,
        useful to inspect what SPLADE considers important.
        """

        pairs = sorted(
            zip(vector["indices"], vector["values"]),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        tokens = self.model.tokenizer.convert_ids_to_tokens(
            [index for index, _ in pairs]
        )

        return [
            (token, value)
            for token, (_, value) in zip(tokens, pairs)
        ]


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

    sparse_model_name:
        Optional SPLADE model. When set, `embed` also fills
        `EmbeddedChunk.sparse_embedding`.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
        sparse_model_name: str | None = None,
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

        self.sparse_embedder = (
            SpladeEmbedder(
                model_name=sparse_model_name,
                batch_size=batch_size,
                device=device,
            )
            if sparse_model_name
            else None
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

        sparse_embeddings = (
            self.sparse_embedder.embed_documents(texts)
            if self.sparse_embedder is not None
            else [None] * len(texts)
        )

        return [
            EmbeddedChunk(
                chunk_id=i,
                text=texts[i],
                embedding=embeddings[i].tolist(),
                source_chunk=chunks[i],
                metadata=self._get_metadata(chunks[i]),
                sparse_embedding=sparse_embeddings[i],
            )
            for i in range(len(chunks))
        ]

    def embed_vectors(
        self,
        chunks: Sequence[Any],
        prompt_name: str | None = None,
    ) -> np.ndarray:
        """
        Return only the embedding matrix.

        Shape:
            (number_of_chunks, embedding_dimension)

        prompt_name:
            Name of a prompt defined by the model (e.g. "query" for
            Qwen3-Embedding). Ignored if the model does not define it.
        """

        texts = [
            self._get_text(chunk)
            for chunk in chunks
        ]

        if prompt_name not in self.model.prompts:
            prompt_name = None

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            prompt_name=prompt_name,
            show_progress_bar=False,
        )
