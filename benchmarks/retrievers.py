"""Uniform retriever wrappers.

Every retriever exposes:

    retrieve(query, paper_id, k) -> RetrievalResult

where `paper_id` scopes the search to a single QASPER paper (matching how
QASPER questions are anchored to one paper at a time), and `k` is a target
size that fixed-k retrievers respect verbatim. Adaptive retrievers (DRAG
beam-knee variants) ignore `k` and return whatever the algorithm picks.

Results are returned as a flat list of leaf paragraph_ids in rank order.
For DRAG methods that return interior tree nodes, we expand each returned
node to its leaf descendants and dedupe by first occurrence — preserving the
"the answer lies in this subtree" semantics for Recall while keeping the
metric machinery method-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    Document,
    FusionQuery,
    Fusion,
    ScoredPoint,
)

from rag_lib.clients import RAGalicClient
from rag_lib.search import (
    branch_search_points,
    beam_search_points,
    DEFAULT_FUSION_SCHEDULE,
)
from rag_lib.utils import llm_call


def _schedule_or_none(weight_schedule: bool) -> list[tuple[float, float]] | None:
    """Translate the user-facing boolean flag into the actual schedule list
    that the search-layer functions expect."""
    return DEFAULT_FUSION_SCHEDULE if weight_schedule else None

RetrieverName = Literal[
    "vanilla_dense",
    "bm25_only",
    "hybrid_rrf_flat",
    "hyde_hybrid",
    "raptor_collapsed",
    "drag_branch",
    "drag_beam_fixed",
    "drag_beam_knee",
    "drag_beam_sensitive_knee",
    "drag_beam_sensk_0.25",
    "drag_beam_sensk_0.5",
    "drag_beam_sensk_0.75",
    "drag_beam_scheduled",
]


@dataclass
class RetrievalResult:
    retriever: RetrieverName
    paragraph_ids: list[str]
    k_returned: int
    latency_s: float
    raw_payloads: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Flat retrievers (vanilla / bm25 / hybrid) over the `bench_flat` collection
# ---------------------------------------------------------------------------


def _paper_filter(paper_id: str | None) -> Filter | None:
    """Build a Qdrant filter that scopes to one paper. None → no filter
    (cross-paper search; the retriever sees every paper's leaves)."""
    if paper_id is None:
        return None
    return Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])


def _flat_query(
    query: str,
    paper_id: str | None,
    k: int,
    mode: Literal["dense", "sparse", "hybrid"],
    collection_name: str,
    dense_text: str | None = None,
) -> list[ScoredPoint]:
    """One Qdrant call, scoped to paper_id (or all papers if None).

    `dense_text` overrides the text used for the dense embedding (HyDE uses
    this to embed a hypothetical answer instead of the raw query). The
    sparse side always uses the raw query.
    """
    f = _paper_filter(paper_id)
    dense_query_text = dense_text if dense_text is not None else query
    with RAGalicClient() as ctx:
        client = ctx.client
        dense_model: str = client.embedding_model_name  # type: ignore
        sparse_model: str = client.sparse_embedding_model_name  # type: ignore
        if mode == "dense":
            return client.query_points(
                collection_name=collection_name,
                query=Document(text=dense_query_text, model=dense_model),
                using="dense",
                query_filter=f,
                limit=k,
            ).points
        if mode == "sparse":
            return client.query_points(
                collection_name=collection_name,
                query=Document(text=query, model=sparse_model),
                using="sparse",
                query_filter=f,
                limit=k,
            ).points
        return client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=Document(text=dense_query_text, model=dense_model),
                    using="dense",
                    limit=k * 3,
                    filter=f,
                ),
                Prefetch(
                    query=Document(text=query, model=sparse_model),
                    using="sparse",
                    limit=k * 3,
                    filter=f,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=f,
            limit=k,
        ).points


def _points_to_paragraph_ids(points: list[ScoredPoint]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in points:
        pid = p.payload.get("paragraph_id") if p.payload else None
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def retrieve_vanilla_dense(
    query: str, paper_id: str | None, k: int, collection_name: str = "bench_flat", **_kw
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = _flat_query(query, paper_id, k, "dense", collection_name)
    elapsed = time.perf_counter() - t0
    pids = _points_to_paragraph_ids(points)
    return RetrievalResult(
        retriever="vanilla_dense",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


def retrieve_bm25_only(
    query: str, paper_id: str | None, k: int, collection_name: str = "bench_flat", **_kw
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = _flat_query(query, paper_id, k, "sparse", collection_name)
    elapsed = time.perf_counter() - t0
    pids = _points_to_paragraph_ids(points)
    return RetrievalResult(
        retriever="bm25_only",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


def retrieve_hybrid_rrf_flat(
    query: str, paper_id: str | None, k: int, collection_name: str = "bench_flat", **_kw
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = _flat_query(query, paper_id, k, "hybrid", collection_name)
    elapsed = time.perf_counter() - t0
    pids = _points_to_paragraph_ids(points)
    return RetrievalResult(
        retriever="hybrid_rrf_flat",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


# ---------------------------------------------------------------------------
# Subtree-expansion helpers (shared by DRAG and RAPTOR)
# ---------------------------------------------------------------------------


def _fetch_subtree_leaves(
    collection_name: str, paper_id: str | None, root_node_ids: list[int]
) -> dict[int, list[str]]:
    """For each given node id (interior or leaf), return its leaf paragraph_ids
    in document order.
    """
    f = _paper_filter(paper_id)
    with RAGalicClient() as ctx:
        client = ctx.client
        # Scroll all nodes for this paper — small per-paper, cheap.
        all_points: list[dict] = []
        offset = None
        while True:
            page, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=f,
                with_payload=True,
                with_vectors=False,
                limit=1024,
                offset=offset,
            )
            all_points.extend([p.payload for p in page if p.payload])
            if offset is None:
                break

    by_id: dict[int, dict] = {p["id"]: p for p in all_points}

    def expand(node_id: int) -> list[dict]:
        node = by_id.get(node_id)
        if node is None:
            return []
        if node.get("is_leaf"):
            return [node]
        out: list[dict] = []
        for cid in node.get("child_ids", []) or []:
            out.extend(expand(cid))
        return out

    result: dict[int, list[str]] = {}
    for root_id in root_node_ids:
        leaves = expand(root_id)
        leaves.sort(key=lambda n: n["page_start"])
        result[root_id] = [
            ln["paragraph_id"] for ln in leaves if ln.get("paragraph_id")
        ]
    return result


def _drag_points_to_paragraph_ids(
    collection_name: str, paper_id: str | None, points: list[ScoredPoint]
) -> list[str]:
    """Convert DRAG/RAPTOR returned points to a ranked, deduped list of leaf
    paragraph_ids by expanding each subtree.
    """
    node_ids = [p.payload["id"] for p in points if p.payload]
    leaves_by_node = _fetch_subtree_leaves(collection_name, paper_id, node_ids)
    seen: set[str] = set()
    out: list[str] = []
    for p in points:
        nid = p.payload["id"] if p.payload else None
        for pid in leaves_by_node.get(nid, []):
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


# ---------------------------------------------------------------------------
# DRAG retrievers (over the `bench_drag` collection)
# ---------------------------------------------------------------------------


def retrieve_drag_branch(
    query: str,
    paper_id: str | None,
    k: int,  # ignored — branch_search picks its own depth
    collection_name: str = "bench_drag",
    num_roots: int = 3,
    weight_schedule: bool = False,
    **_kw,
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = branch_search_points(
        query=query,
        num_roots=num_roots,
        collection_name=collection_name,
        extra_root_filter=_paper_filter(paper_id),
        schedule=_schedule_or_none(weight_schedule),
    )
    elapsed = time.perf_counter() - t0
    pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
    return RetrievalResult(
        retriever="drag_branch",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


def retrieve_drag_beam_fixed(
    query: str,
    paper_id: str | None,
    k: int,
    collection_name: str = "bench_drag",
    weight_schedule: bool = False,
    **_kw,
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = beam_search_points(
        query=query,
        beam_width=k,
        search_method="fixed",
        collection_name=collection_name,
        extra_root_filter=_paper_filter(paper_id),
        schedule=_schedule_or_none(weight_schedule),
    )
    elapsed = time.perf_counter() - t0
    pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
    return RetrievalResult(
        retriever="drag_beam_fixed",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


def retrieve_drag_beam_knee(
    query: str,
    paper_id: str | None,
    k: int,  # ignored — knee picks adaptively
    collection_name: str = "bench_drag",
    max_num_roots: int = 20,
    weight_schedule: bool = False,
    **_kw,
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = beam_search_points(
        query=query,
        search_method="adaptive_with_knee",
        max_num_roots=max_num_roots,
        collection_name=collection_name,
        extra_root_filter=_paper_filter(paper_id),
        schedule=_schedule_or_none(weight_schedule),
    )
    elapsed = time.perf_counter() - t0
    pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
    return RetrievalResult(
        retriever="drag_beam_knee",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


def retrieve_drag_beam_sensitive_knee(
    query: str,
    paper_id: str | None,
    k: int,
    collection_name: str = "bench_drag",
    sensitivity: float = 0.85,
    max_num_roots: int = 20,
    weight_schedule: bool = False,
    **_kw,
) -> RetrievalResult:
    t0 = time.perf_counter()
    points = beam_search_points(
        query=query,
        search_method="adaptive_with_sensitive_knee",
        sensitivity=sensitivity,
        max_num_roots=max_num_roots,
        collection_name=collection_name,
        extra_root_filter=_paper_filter(paper_id),
        schedule=_schedule_or_none(weight_schedule),
    )
    elapsed = time.perf_counter() - t0
    pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
    return RetrievalResult(
        retriever="drag_beam_sensitive_knee",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


# ---------------------------------------------------------------------------
# RAPTOR retriever (over `bench_raptor`)
# ---------------------------------------------------------------------------


def retrieve_raptor_collapsed(
    query: str, paper_id: str | None, k: int, collection_name: str = "bench_raptor", **_kw
) -> RetrievalResult:
    """RAPTOR's 'collapsed tree' retrieval: rank all nodes (leaves + cluster
    summaries) jointly by hybrid score, then return the top-k, expanded to
    their leaf paragraph_ids.
    """
    f = _paper_filter(paper_id)
    t0 = time.perf_counter()
    with RAGalicClient() as ctx:
        client = ctx.client
        dense_model: str = client.embedding_model_name  # type: ignore
        sparse_model: str = client.sparse_embedding_model_name  # type: ignore
        points = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=dense_model),
                    using="dense",
                    limit=k * 3,
                    filter=f,
                ),
                Prefetch(
                    query=Document(text=query, model=sparse_model),
                    using="sparse",
                    limit=k * 3,
                    filter=f,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=f,
            limit=k,
        ).points
    elapsed = time.perf_counter() - t0
    pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
    return RetrievalResult(
        retriever="raptor_collapsed",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


# ---------------------------------------------------------------------------
# HyDE (Hypothetical Document Embeddings, Gao et al. 2022)
# ---------------------------------------------------------------------------

HYDE_SYSTEM_PROMPT = (
    "You are writing a short, specific passage that would plausibly appear in "
    "a research paper as the answer to the user's question. Invent reasonable "
    "specifics (numbers, dataset names, method names) if needed. Do not hedge, "
    "do not refuse, do not say 'I don't know'. Output the passage only — no "
    "preamble, no 'Here is...'. 2-4 sentences."
)


def _generate_hyde(query: str) -> str:
    """Generate a hypothetical answer passage to use as the dense query.

    Gao et al. 2022: 'Precise Zero-Shot Dense Retrieval without Relevance
    Labels'. The intuition is that the LLM-written answer occupies the same
    embedding subspace as real document passages, so cosine similarity in
    that space is more discriminative than query↔passage similarity.
    """
    messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        content, _ = llm_call(messages)
        passage = (content or "").strip()
        if not passage:
            return query
        return passage
    except Exception:
        return query


def retrieve_hyde_hybrid(
    query: str, paper_id: str | None, k: int, collection_name: str = "bench_flat", **_kw
) -> RetrievalResult:
    """HyDE + hybrid RRF: dense side embeds the hypothetical answer, sparse
    side keeps the raw query (BM25 needs the actual query keywords)."""
    t0 = time.perf_counter()
    hyde_passage = _generate_hyde(query)
    points = _flat_query(
        query=query,
        paper_id=paper_id,
        k=k,
        mode="hybrid",
        collection_name=collection_name,
        dense_text=hyde_passage,
    )
    elapsed = time.perf_counter() - t0
    pids = _points_to_paragraph_ids(points)
    return RetrievalResult(
        retriever="hyde_hybrid",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


# ---------------------------------------------------------------------------
# Sensitivity sweep: sensitive_knee at multiple thresholds (0.85 already
# registered as drag_beam_sensitive_knee; below add 0.25 / 0.5 / 0.75)
# ---------------------------------------------------------------------------


def _make_sensk_retriever(sensitivity: float, name: str):
    def retrieve(
        query: str,
        paper_id: str | None,
        k: int,
        collection_name: str = "bench_drag",
        max_num_roots: int = 20,
        weight_schedule: bool = False,
        **_kw,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        points = beam_search_points(
            query=query,
            search_method="adaptive_with_sensitive_knee",
            sensitivity=sensitivity,
            max_num_roots=max_num_roots,
            collection_name=collection_name,
            extra_root_filter=_paper_filter(paper_id),
            schedule=_schedule_or_none(weight_schedule),
        )
        elapsed = time.perf_counter() - t0
        pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
        return RetrievalResult(
            retriever=name,  # type: ignore[arg-type]
            paragraph_ids=pids,
            k_returned=len(pids),
            latency_s=elapsed,
            raw_payloads=[p.payload for p in points],
        )

    retrieve.__name__ = f"retrieve_{name}"
    return retrieve


retrieve_drag_beam_sensk_025 = _make_sensk_retriever(0.25, "drag_beam_sensk_0.25")
retrieve_drag_beam_sensk_050 = _make_sensk_retriever(0.5, "drag_beam_sensk_0.5")
retrieve_drag_beam_sensk_075 = _make_sensk_retriever(0.75, "drag_beam_sensk_0.75")


# ---------------------------------------------------------------------------
# Scheduled-fusion beam: BM25-weighted at root, dense-weighted at depth
# ---------------------------------------------------------------------------


def retrieve_drag_beam_scheduled(
    query: str,
    paper_id: str | None,
    k: int,
    collection_name: str = "bench_drag",
    weight_schedule: bool = True,  # this retriever is the schedule by definition
    **_kw,
) -> RetrievalResult:
    """Hypothesis: at the document/section level (early beam steps), BM25
    keyword matching identifies the right topic; at the leaf level (late
    steps), dense semantic similarity identifies the right fact.

    This wrapper drives the scheduled-weight beam exposed by
    `rag_lib.search.beam_search_points_scheduled`.
    """
    from rag_lib.search import beam_search_points_scheduled

    schedule = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2), (1.0, 0.0)]
    t0 = time.perf_counter()
    points = beam_search_points_scheduled(
        query=query,
        schedule=schedule,
        beam_width=k,
        collection_name=collection_name,
        extra_root_filter=_paper_filter(paper_id),
    )
    elapsed = time.perf_counter() - t0
    pids = _drag_points_to_paragraph_ids(collection_name, paper_id, points)
    return RetrievalResult(
        retriever="drag_beam_scheduled",
        paragraph_ids=pids,
        k_returned=len(pids),
        latency_s=elapsed,
        raw_payloads=[p.payload for p in points],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RETRIEVERS = {
    "vanilla_dense": retrieve_vanilla_dense,
    "bm25_only": retrieve_bm25_only,
    "hybrid_rrf_flat": retrieve_hybrid_rrf_flat,
    "hyde_hybrid": retrieve_hyde_hybrid,
    "raptor_collapsed": retrieve_raptor_collapsed,
    "drag_branch": retrieve_drag_branch,
    "drag_beam_fixed": retrieve_drag_beam_fixed,
    "drag_beam_knee": retrieve_drag_beam_knee,
    "drag_beam_sensitive_knee": retrieve_drag_beam_sensitive_knee,
    "drag_beam_sensk_0.25": retrieve_drag_beam_sensk_025,
    "drag_beam_sensk_0.5": retrieve_drag_beam_sensk_050,
    "drag_beam_sensk_0.75": retrieve_drag_beam_sensk_075,
    "drag_beam_scheduled": retrieve_drag_beam_scheduled,
}


def collection_for(retriever: str) -> str:
    if retriever == "raptor_collapsed":
        return "bench_raptor"
    if retriever.startswith("drag_"):
        return "bench_drag"
    return "bench_flat"
