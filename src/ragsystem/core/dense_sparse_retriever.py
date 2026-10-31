"""
Dense + sparse retriever ("Embedding + sparse").

Best method in the retrieval evaluation (see docs/rag_evaluation.md):
every chunk is scored with

    - the Qwen3 dense embedding of the chunk text   (cosine similarity)
    - the SPLADE sparse vector of the chunk text    (dot product)

each score is min-max normalized over the collection, and the two are
averaged with equal weights.

Ranking uses the "Common first" fusion (best chunk@5 in the evaluation):
each signal takes its own top `agreement_top_n` chunks; chunks found by
both signals come first, then chunks found by one signal, then the rest.
Inside each group, chunks are ordered by the averaged score.

All vectors are loaded in memory at startup, which is fine for a few
tens of thousands of chunks.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np
from scipy.sparse import csr_matrix

from ragsystem.core.rag_pipeline.vectorizer import (
    DoclingChunkEmbedder,
    SpladeEmbedder,
)


SPARSE_EMBEDDING_KEY = "sparse_embedding"


@dataclass
class ScoredChunk:
    """
    A chunk returned by the retriever.
    """

    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    dense_score: float
    sparse_score: float


def sparse_matrix(
    vectors: list[dict | None],
    dimension: int,
) -> csr_matrix:
    """
    Stack {"indices", "values"} dicts into a (n, dimension) CSR matrix.

    Missing vectors become empty rows (score 0).
    """

    rows, cols, values = [], [], []

    for row, vector in enumerate(vectors):
        if not vector:
            continue

        rows.extend([row] * len(vector["indices"]))
        cols.extend(vector["indices"])
        values.extend(vector["values"])

    return csr_matrix(
        (values, (rows, cols)),
        shape=(len(vectors), dimension),
        dtype=np.float32,
    )


def min_max(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a score vector to [0, 1].
    """

    low, high = scores.min(), scores.max()

    if high - low <= 0:
        return np.zeros_like(scores)

    return (scores - low) / (high - low)


class DenseSparseRetriever:

    def __init__(
        self,
        chroma_dir: str = "./chroma_db_v1",
        collection_name: str = "documents",
        dense_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        sparse_model_name: str = "naver/splade-cocondenser-selfdistil",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        agreement_top_n: int | None = 10,
    ):
        if dense_weight < 0 or sparse_weight < 0 or dense_weight + sparse_weight == 0:
            raise ValueError("Weights must be >= 0 and not both 0.")

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # None = plain score average, without "common first".
        self.agreement_top_n = agreement_top_n

        # Models are not guaranteed to be thread-safe.
        self._lock = threading.Lock()

        # ---------------------------------------------------------
        # Models
        # ---------------------------------------------------------

        self.dense = DoclingChunkEmbedder(
            model_name=dense_model_name,
            normalize_embeddings=True,
        )
        self.splade = SpladeEmbedder(model_name=sparse_model_name)

        # SPLADE indices are token ids of the tokenizer vocabulary.
        self.vocabulary_size = len(self.splade.model.tokenizer)

        # ---------------------------------------------------------
        # Chunks
        # ---------------------------------------------------------

        collection = chromadb.PersistentClient(path=chroma_dir).get_collection(
            name=collection_name
        )

        data = collection.get(include=["documents", "metadatas", "embeddings"])

        self.ids: list[str] = data["ids"]
        self.documents: list[str] = data["documents"]
        self.metadatas: list[dict] = [m or {} for m in data["metadatas"]]

        # Stored embeddings are L2-normalized: dot product = cosine.
        self.dense_matrix = np.asarray(data["embeddings"], dtype=np.float32)

        self.sparse_matrix = sparse_matrix(
            [
                json.loads(m[SPARSE_EMBEDDING_KEY]) if m.get(SPARSE_EMBEDDING_KEY) else None
                for m in self.metadatas
            ],
            self.vocabulary_size,
        )

        self.missing_sparse = sum(
            1 for m in self.metadatas if not m.get(SPARSE_EMBEDDING_KEY)
        )

    @property
    def size(self) -> int:
        return len(self.ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredChunk]:

        if not query or not query.strip() or not self.ids:
            return []

        with self._lock:
            # Same query encoding as in the evaluation.
            query_dense = np.asarray(
                self.dense.embed_vectors([query], prompt_name="query"),
                dtype=np.float32,
            )[0]
            query_sparse = self.splade.embed_query(query)

        dense_scores = self.dense_matrix @ query_dense

        query_vector = sparse_matrix([query_sparse], self.vocabulary_size)
        sparse_scores = (self.sparse_matrix @ query_vector.T).toarray().ravel()

        total = self.dense_weight + self.sparse_weight
        scores = (
            self.dense_weight * min_max(dense_scores)
            + self.sparse_weight * min_max(sparse_scores)
        ) / total

        if self.agreement_top_n:
            n = min(self.agreement_top_n, len(scores))
            votes = np.zeros_like(scores)
            for signal in (dense_scores, sparse_scores):
                votes[np.argpartition(-signal, n - 1)[:n]] += 1

            # The averaged score is in [0, 1]: it never outweighs one vote.
            ranking = 2.0 * votes + scores
        else:
            ranking = scores

        top = np.argsort(-ranking)[:top_k]

        return [
            ScoredChunk(
                id=self.ids[index],
                text=self.documents[index],
                metadata=self.metadatas[index],
                score=float(scores[index]),
                dense_score=float(dense_scores[index]),
                sparse_score=float(sparse_scores[index]),
            )
            for index in top
        ]
