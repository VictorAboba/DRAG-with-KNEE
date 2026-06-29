"""Question-expansion isolation experiment.

Builds a single leaf-only collection `bench_qexp` with THREE named vectors
per point so we can compare each retrieval signal in isolation:

    dense              dense embedding of the raw paragraph text
    sparse_bullets     BM25 embedding of just the LLM-extracted bullets
    sparse_questions   BM25 embedding of just the LLM-anticipated questions

This separates the question-expansion hypothesis from the keyword-extraction
hypothesis: can a question-to-question match alone beat both dense-content
and keyword-BM25 on QASPER, where users genuinely write questions?

Indexed leaves only — no tree. Question-expansion is a leaf-level signal;
testing it cleanly means cutting the hierarchy noise out.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Iterable

from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVectorParams,
    Modifier,
)

from rag_lib.clients import RAGalicClient

from .datasets import Paper
from .indexing import IndexingStats, _node_uuid
from .rich_descriptor import describe_leaf_bullets_only, LLMSummarizer


QEXP_COLLECTION = "bench_qexp"


def _ensure_qexp_collection(client_ctx, collection_name: str) -> None:
    if client_ctx.client.collection_exists(collection_name):
        return
    # Single dense vector + two named sparse vectors. Same dense model and
    # sparse model the rest of the benchmark uses, so retrieval quality is
    # directly comparable to bench_flat.
    dense_params = list(
        client_ctx.client.get_fastembed_vector_params().items()
    )[0][1]
    sparse_default = list(
        client_ctx.client.get_fastembed_sparse_vector_params().items()
    )[0][1]
    sparse_params = SparseVectorParams(modifier=Modifier.IDF)
    client_ctx.client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": dense_params},
        sparse_vectors_config={
            "sparse_bullets": sparse_default,
            "sparse_questions": sparse_params,
        },
    )


def index_qexp_leaves(
    papers: Iterable[Paper],
    collection_name: str = QEXP_COLLECTION,
    summarizer: LLMSummarizer | None = None,
) -> IndexingStats:
    """One LLM call per paragraph -> bullets + anticipated_questions.

    Each leaf is upserted with three vectors:
      dense from para.text
      sparse_bullets from " | ".join(bullets) (or empty -> skip vector)
      sparse_questions from " | ".join(anticipated_questions) (or skip)

    Skip-on-empty means a paragraph with no bullets at all won't get a
    sparse_bullets vector — Qdrant treats absent named vectors fine; the
    point just won't surface in that retriever, which is the right
    semantic.
    """
    from rag_lib.utils import llm_call as default_llm

    if summarizer is None:
        summarizer = default_llm

    stats = IndexingStats()
    t0 = time.perf_counter()

    with RAGalicClient() as client_ctx:
        _ensure_qexp_collection(client_ctx, collection_name)
        client = client_ctx.client
        dense_model: str = client.embedding_model_name  # type: ignore
        sparse_model: str = client.sparse_embedding_model_name  # type: ignore

        papers_list = list(papers)
        n_papers = len(papers_list)
        node_id = 0

        for pi, paper in enumerate(papers_list, 1):
            n_paras = len(paper.paragraphs)
            print(
                f"[qexp_index] paper {pi}/{n_papers} {paper.paper_id}: "
                f"{n_paras} paragraphs",
                flush=True,
            )
            leaf_t0 = time.perf_counter()
            batch: list[PointStruct] = []
            for li, para in enumerate(paper.paragraphs, 1):
                desc = describe_leaf_bullets_only(para, summarizer)
                stats.llm_calls += 1
                if li % 5 == 0 or li == n_paras:
                    rate = li / max(time.perf_counter() - leaf_t0, 1e-6)
                    print(
                        f"[qexp_index]   leaves {li}/{n_paras} "
                        f"({rate:.2f}/s, llm_calls={stats.llm_calls})",
                        flush=True,
                    )
                bullets_text = " | ".join(desc.bullets).strip()
                questions_text = " | ".join(desc.anticipated_questions).strip()

                payload = {
                    "id": node_id,
                    "paper_id": paper.paper_id,
                    "paragraph_id": para.paragraph_id,
                    "text": para.text,
                    "bullets": desc.bullets,
                    "anticipated_questions": desc.anticipated_questions,
                    "is_leaf": True,
                }
                vector: dict = {
                    "dense": Document(text=para.text, model=dense_model),
                }
                if bullets_text:
                    vector["sparse_bullets"] = Document(
                        text=bullets_text, model=sparse_model
                    )
                if questions_text:
                    vector["sparse_questions"] = Document(
                        text=questions_text, model=sparse_model
                    )
                batch.append(
                    PointStruct(
                        id=_node_uuid(collection_name, node_id),
                        vector=vector,
                        payload=payload,
                    )
                )
                node_id += 1
                stats.leaves += 1
                # Flush every 8 to keep memory bounded and progress visible.
                if len(batch) >= 8:
                    client.upsert(
                        collection_name=collection_name, points=batch, wait=True
                    )
                    batch.clear()
            if batch:
                client.upsert(
                    collection_name=collection_name, points=batch, wait=True
                )

    stats.seconds = time.perf_counter() - t0
    return stats


def drop_qexp_collection(collection_name: str = QEXP_COLLECTION) -> None:
    with RAGalicClient() as client_ctx:
        if client_ctx.client.collection_exists(collection_name):
            client_ctx.client.delete_collection(collection_name)
