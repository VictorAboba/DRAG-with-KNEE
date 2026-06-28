"""Populate the three Qdrant collections backing the benchmark.

We keep one collection per indexing strategy so each retriever has the data
shape it expects:

    bench_flat    leaves only, parent_id=-1, child_ids=[]
                  vanilla_dense / bm25_only / hybrid_rrf_flat query this
    bench_drag    hierarchical tree (leaves + parents + root) built the same
                  way build_tree.build_tree does, with LLM descriptions
    bench_raptor  hierarchical tree built via recursive clustering +
                  LLM-generated cluster summaries (RAPTOR-style)

All three share the same QASPER paragraphs as leaves so retrieval is comparable.
Each leaf carries a `paragraph_id` payload that the metrics module matches
against the gold-evidence IDs from `datasets.Question.gold_paragraph_ids`.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from qdrant_client.models import PointStruct, Document

from rag_lib.clients import RAGalicClient
from rag_lib.dataschemes import DescriptorOutput
from rag_lib.utils import llm_call
from rag_lib.build_tree import DESCRIPTOR_SYSTEM_PROMPT

from .datasets import Paper, Paragraph
from .rich_descriptor import (
    describe_leaf_rich,
    describe_parent_rich,
    describe_parent_with_bullets,
)


# Signature matches rag_lib.utils.llm_call: (messages, structured_output) -> (content, reasoning)
LLMSummarizer = Callable[..., tuple[str, str]]


@dataclass
class IndexingStats:
    leaves: int = 0
    parent_nodes: int = 0
    llm_calls: int = 0
    seconds: float = 0.0


def _node_uuid(collection: str, node_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection}::{node_id}"))


def _build_payload(
    *,
    node_id: int,
    paper_id: str,
    parent_id: int,
    child_ids: list[int],
    description: str,
    keywords: list[str],
    page_start: int,
    page_end: int,
    paragraph_id: str | None,
    text: str,
    is_leaf: bool,
    level: int,
) -> dict:
    return {
        "id": node_id,
        "file_name": paper_id,
        "paper_id": paper_id,
        "parent_id": parent_id,
        "child_ids": child_ids,
        "description": description,
        "keywords": keywords,
        "page_start": page_start,
        "page_end": page_end,
        "paragraph_id": paragraph_id,
        "text": text,
        "is_leaf": is_leaf,
        "level": level,
    }


def _ensure_collection(client_ctx, collection_name: str) -> None:
    if not client_ctx.client.collection_exists(collection_name):
        client_ctx.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": list(
                    client_ctx.client.get_fastembed_vector_params().items()
                )[0][1]
            },
            sparse_vectors_config={
                "sparse": list(
                    client_ctx.client.get_fastembed_sparse_vector_params().items()
                )[0][1]
            },
        )


def _truncate(text: str, max_chars: int = 1500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _describe_leaf(
    paragraph: Paragraph, summarizer: LLMSummarizer | None
) -> DescriptorOutput:
    if summarizer is None:
        return DescriptorOutput(
            description=_truncate(paragraph.text), keywords=[]
        )
    messages = [
        {"role": "system", "content": DESCRIPTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Raw Page Content:\n{paragraph.text}"},
    ]
    for _ in range(3):
        try:
            output_str, _ = summarizer(messages, DescriptorOutput)
            return DescriptorOutput.model_validate_json(output_str)
        except Exception:
            continue
    return DescriptorOutput(description=_truncate(paragraph.text), keywords=[])


def _describe_parent(
    child_descriptions: list[tuple[str, list[str]]],
    summarizer: LLMSummarizer | None,
) -> DescriptorOutput:
    if summarizer is None:
        merged = " | ".join(desc for desc, _ in child_descriptions)
        merged_kw: list[str] = sorted(
            {kw for _, kws in child_descriptions for kw in kws}
        )
        return DescriptorOutput(description=_truncate(merged), keywords=merged_kw)
    combined = "\n\n".join(
        f"Child #{i}\nDescription: {desc}\nKeywords: {', '.join(kws)}"
        for i, (desc, kws) in enumerate(child_descriptions, 1)
    )
    messages = [
        {"role": "system", "content": DESCRIPTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Child Metadata:\n{combined}"},
    ]
    for _ in range(3):
        try:
            output_str, _ = summarizer(messages, DescriptorOutput)
            return DescriptorOutput.model_validate_json(output_str)
        except Exception:
            continue
    merged = " | ".join(desc for desc, _ in child_descriptions)
    return DescriptorOutput(description=_truncate(merged), keywords=[])


def _dense_text(description: str | None) -> str:
    return description or "N/A"


def _sparse_text(file_name: str, description: str | None, keywords: list[str]) -> str:
    return (
        f"{file_name}, {description or 'N/A'}, "
        f"{', '.join(keywords) or 'N/A'}"
    )


def _upsert(client_ctx, collection: str, payloads: list[dict]) -> None:
    dense_model = client_ctx.client.embedding_model_name  # type: ignore
    sparse_model = client_ctx.client.sparse_embedding_model_name  # type: ignore
    points = [
        PointStruct(
            id=_node_uuid(collection, p["id"]),
            vector={
                "dense": Document(text=_dense_text(p["description"]), model=dense_model),
                "sparse": Document(
                    text=_sparse_text(p["file_name"], p["description"], p["keywords"]),
                    model=sparse_model,
                ),
            },
            payload=p,
        )
        for p in payloads
    ]
    client_ctx.client.upsert(collection_name=collection, points=points, wait=True)


def index_flat_leaves(
    papers: Iterable[Paper],
    collection_name: str = "bench_flat",
    skip_paper_ids: set[str] | None = None,
    start_node_id: int = 0,
) -> IndexingStats:
    """Flat indexing: every paragraph becomes a parent_id=-1 leaf-root.

    No LLM calls. Embedding is over the raw paragraph text. Vanilla dense,
    BM25-only, and hybrid_rrf_flat all share this collection.

    `skip_paper_ids` causes papers whose id is already in the collection to
    be skipped (used by `--incremental`). `start_node_id` offsets the local
    node-id counter so new points don't collide with previously-indexed ones.
    """
    stats = IndexingStats()
    start = time.perf_counter()
    payloads: list[dict] = []
    node_id = start_node_id
    skip_paper_ids = skip_paper_ids or set()
    for paper in papers:
        if paper.paper_id in skip_paper_ids:
            print(
                f"[index_flat]   skipping {paper.paper_id} (already indexed)",
                flush=True,
            )
            continue
        for para in paper.paragraphs:
            payloads.append(
                _build_payload(
                    node_id=node_id,
                    paper_id=paper.paper_id,
                    parent_id=-1,
                    child_ids=[],
                    description=para.text,
                    keywords=[],
                    page_start=para.global_idx,
                    page_end=para.global_idx,
                    paragraph_id=para.paragraph_id,
                    text=para.text,
                    is_leaf=True,
                    level=0,
                )
            )
            node_id += 1
            stats.leaves += 1

    with RAGalicClient() as client_ctx:
        _ensure_collection(client_ctx, collection_name)
        _flush_in_batches(client_ctx, collection_name, payloads, batch_size=8)
    stats.seconds = time.perf_counter() - start
    return stats


def _flush_in_batches(
    client_ctx, collection: str, payloads: list[dict], batch_size: int
) -> None:
    for i in range(0, len(payloads), batch_size):
        _upsert(client_ctx, collection, payloads[i : i + batch_size])


def index_drag_tree(
    papers: Iterable[Paper],
    collection_name: str = "bench_drag",
    width: int = 3,
    summarizer: LLMSummarizer | None = llm_call,
    skip_paper_ids: set[str] | None = None,
    start_node_id: int = 0,
) -> IndexingStats:
    """Build the DRAG hierarchical tree exactly like rag_lib.build_tree, but
    fed by pre-chunked QASPER paragraphs instead of PDF parsing.

    Each paragraph -> LLM-described leaf. Then groups of `width` leaves are
    aggregated into a parent node (with its own LLM description), recursively
    until one root per paper.

    `skip_paper_ids` causes papers whose id is already in the collection to
    be skipped (used by `--incremental`). `start_node_id` offsets the local
    node-id counter so new points don't collide with previously-indexed ones.
    """
    stats = IndexingStats()
    start = time.perf_counter()
    all_payloads: list[dict] = []
    node_id = start_node_id
    skip_paper_ids = skip_paper_ids or set()

    papers_list = [p for p in papers if p.paper_id not in skip_paper_ids]
    skipped = [p.paper_id for p in papers if p.paper_id in skip_paper_ids]
    for sid in skipped:
        print(f"[index_drag]   skipping {sid} (already indexed)", flush=True)
    n_papers = len(papers_list)
    for pi, paper in enumerate(papers_list, 1):
        n_paras = len(paper.paragraphs)
        print(
            f"[index_drag] paper {pi}/{n_papers} {paper.paper_id}: "
            f"{n_paras} paragraphs",
            flush=True,
        )
        # Leaves — rich (abstract + bullets) descriptions, stored in
        # the existing `description` / `keywords` payload fields so the
        # rest of the pipeline (`_dense_text`, `_sparse_text`, search.py)
        # needs no changes.
        leaf_payloads: list[dict] = []
        leaf_t0 = time.perf_counter()
        for li, para in enumerate(paper.paragraphs, 1):
            desc = describe_leaf_rich(para, summarizer)
            if summarizer is not None:
                stats.llm_calls += 1
            if li % 5 == 0 or li == n_paras:
                rate = li / max(time.perf_counter() - leaf_t0, 1e-6)
                print(
                    f"[index_drag]   leaves {li}/{n_paras} ({rate:.1f}/s, "
                    f"llm_calls={stats.llm_calls})",
                    flush=True,
                )
            leaf_payloads.append(
                _build_payload(
                    node_id=node_id,
                    paper_id=paper.paper_id,
                    parent_id=-2,  # placeholder, fixed up below
                    child_ids=[],
                    description=desc.abstract,
                    keywords=desc.bullets,
                    page_start=para.global_idx,
                    page_end=para.global_idx,
                    paragraph_id=para.paragraph_id,
                    text=para.text,
                    is_leaf=True,
                    level=0,
                )
            )
            node_id += 1
            stats.leaves += 1

        # Aggregate up by `width` until single root
        all_payloads.extend(leaf_payloads)
        current_level = leaf_payloads
        level = 1
        while len(current_level) > 1:
            new_level: list[dict] = []
            print(
                f"[index_drag]   building level {level}: {len(current_level)} -> ~{(len(current_level)+width-1)//width}",
                flush=True,
            )
            for i in range(0, len(current_level), width):
                group = current_level[i : i + width]
                if len(group) == 1:
                    new_level.append(group[0])
                    continue
                child_pairs = [(g["description"] or "", g["keywords"]) for g in group]
                desc = describe_parent_rich(child_pairs, summarizer)
                if summarizer is not None:
                    stats.llm_calls += 1
                parent_payload = _build_payload(
                    node_id=node_id,
                    paper_id=paper.paper_id,
                    parent_id=-2,
                    child_ids=[g["id"] for g in group],
                    description=desc.abstract,
                    keywords=desc.bullets or sorted({kw for _, kws in child_pairs for kw in kws}),
                    page_start=min(g["page_start"] for g in group),
                    page_end=max(g["page_end"] for g in group),
                    paragraph_id=None,
                    text="",
                    is_leaf=False,
                    level=level,
                )
                for g in group:
                    g["parent_id"] = parent_payload["id"]
                node_id += 1
                stats.parent_nodes += 1
                all_payloads.append(parent_payload)
                new_level.append(parent_payload)
            current_level = new_level
            level += 1

        # Root: mark parent_id = -1
        if current_level:
            current_level[0]["parent_id"] = -1

    # Any payload still on the -2 placeholder is also a (single-leaf) root
    for p in all_payloads:
        if p["parent_id"] == -2:
            p["parent_id"] = -1

    with RAGalicClient() as client_ctx:
        _ensure_collection(client_ctx, collection_name)
        _flush_in_batches(client_ctx, collection_name, all_payloads, batch_size=8)

    stats.seconds = time.perf_counter() - start
    return stats


def _embed_leaves_dense(paragraphs: list[Paragraph]) -> np.ndarray:
    """Embed paragraphs via the same dense model used for retrieval.

    Used by RAPTOR for clustering. Goes through qdrant-fastembed so the
    vectors are identical to what the retriever will compare against.
    """
    from fastembed import TextEmbedding

    embedder = TextEmbedding("jinaai/jina-embeddings-v3")
    texts = [p.text for p in paragraphs]
    vectors = list(embedder.embed(texts))
    return np.vstack(vectors)


def _gmm_cluster(vectors: np.ndarray, max_k: int) -> np.ndarray:
    """RAPTOR-style soft clustering. Picks k via BIC over GMMs.

    Returns a hard assignment vector (length = #vectors). For tiny corpora we
    fall back to a single cluster.
    """
    n = len(vectors)
    if n <= 2:
        return np.zeros(n, dtype=int)
    from sklearn.mixture import GaussianMixture

    candidates = list(range(2, min(max_k, n) + 1))
    if not candidates:
        return np.zeros(n, dtype=int)

    best_k, best_bic, best_labels = 1, float("inf"), np.zeros(n, dtype=int)
    for k in candidates:
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag", random_state=0, reg_covar=1e-4
        )
        try:
            gmm.fit(vectors)
            bic = gmm.bic(vectors)
        except Exception:
            continue
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = gmm.predict(vectors)
    return best_labels


def index_raptor_tree(
    papers: Iterable[Paper],
    collection_name: str = "bench_raptor",
    shrink_ratio: int = 3,
    summarizer: LLMSummarizer | None = llm_call,
    skip_paper_ids: set[str] | None = None,
    start_node_id: int = 0,
) -> IndexingStats:
    """RAPTOR-style hierarchical tree.

    Leaves carry raw paragraph text (no per-leaf LLM call — original RAPTOR
    only summarizes cluster nodes). At each level we GMM-cluster the current
    nodes by their dense embedding, summarize each cluster with the LLM, and
    repeat until the level has <= 1 node.

    `shrink_ratio` bounds the cluster count per level: k_max = ceil(n / shrink_ratio).

    `skip_paper_ids` causes papers whose id is already in the collection to
    be skipped (used by `--incremental`). `start_node_id` offsets the local
    node-id counter so new points don't collide with previously-indexed ones.
    """
    stats = IndexingStats()
    start = time.perf_counter()
    all_payloads: list[dict] = []
    node_id = start_node_id
    skip_paper_ids = skip_paper_ids or set()

    papers_list = [p for p in papers if p.paper_id not in skip_paper_ids]
    skipped = [p.paper_id for p in papers if p.paper_id in skip_paper_ids]
    for sid in skipped:
        print(f"[index_raptor]   skipping {sid} (already indexed)", flush=True)
    n_papers = len(papers_list)
    for pi, paper in enumerate(papers_list, 1):
        n_paras = len(paper.paragraphs)
        print(
            f"[index_raptor] paper {pi}/{n_papers} {paper.paper_id}: "
            f"{n_paras} paragraphs",
            flush=True,
        )
        leaves: list[dict] = []
        for para in paper.paragraphs:
            leaves.append(
                _build_payload(
                    node_id=node_id,
                    paper_id=paper.paper_id,
                    parent_id=-2,
                    child_ids=[],
                    description=para.text,
                    keywords=[],
                    page_start=para.global_idx,
                    page_end=para.global_idx,
                    paragraph_id=para.paragraph_id,
                    text=para.text,
                    is_leaf=True,
                    level=0,
                )
            )
            node_id += 1
            stats.leaves += 1
        all_payloads.extend(leaves)

        current_level_payloads = leaves
        current_level_texts = [p["text"] or p["description"] for p in leaves]
        level = 1

        while len(current_level_payloads) > 1:
            # Embed current-level nodes for clustering
            print(
                f"[index_raptor]   level {level}: clustering {len(current_level_payloads)} nodes",
                flush=True,
            )
            vectors = _embed_text_list(current_level_texts)
            n = len(current_level_payloads)
            k_max = max(1, math.ceil(n / shrink_ratio))
            labels = _gmm_cluster(vectors, k_max)

            new_level_payloads: list[dict] = []
            new_level_texts: list[str] = []
            for cluster_id in sorted(set(labels.tolist())):
                members_idx = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                if not members_idx:
                    continue
                if len(members_idx) == 1:
                    new_level_payloads.append(current_level_payloads[members_idx[0]])
                    new_level_texts.append(current_level_texts[members_idx[0]])
                    continue
                members = [current_level_payloads[i] for i in members_idx]
                child_pairs = [(m["description"] or "", m["keywords"]) for m in members]
                # RAPTOR leaves are raw text with no bullets, so the LLM has
                # to extract bullets at the parent level — not just abstract.
                desc = describe_parent_with_bullets(child_pairs, summarizer)
                if summarizer is not None:
                    stats.llm_calls += 1
                parent_payload = _build_payload(
                    node_id=node_id,
                    paper_id=paper.paper_id,
                    parent_id=-2,
                    child_ids=[m["id"] for m in members],
                    description=desc.abstract,
                    keywords=desc.bullets or sorted({kw for _, kws in child_pairs for kw in kws}),
                    page_start=min(m["page_start"] for m in members),
                    page_end=max(m["page_end"] for m in members),
                    paragraph_id=None,
                    text="",
                    is_leaf=False,
                    level=level,
                )
                for m in members:
                    m["parent_id"] = parent_payload["id"]
                node_id += 1
                stats.parent_nodes += 1
                all_payloads.append(parent_payload)
                new_level_payloads.append(parent_payload)
                new_level_texts.append(parent_payload["description"] or "")

            if len(new_level_payloads) == len(current_level_payloads):
                # No collapse happened; force a wrap-up to avoid infinite loop
                break
            current_level_payloads = new_level_payloads
            current_level_texts = new_level_texts
            level += 1

        if current_level_payloads:
            current_level_payloads[0]["parent_id"] = -1

    for p in all_payloads:
        if p["parent_id"] == -2:
            p["parent_id"] = -1

    with RAGalicClient() as client_ctx:
        _ensure_collection(client_ctx, collection_name)
        _flush_in_batches(client_ctx, collection_name, all_payloads, batch_size=8)

    stats.seconds = time.perf_counter() - start
    return stats


def _embed_text_list(texts: list[str]) -> np.ndarray:
    """Single-source-of-truth embedding helper for clustering."""
    from fastembed import TextEmbedding

    embedder = TextEmbedding("jinaai/jina-embeddings-v3")
    return np.vstack(list(embedder.embed(texts)))


def drop_collection(collection_name: str) -> None:
    with RAGalicClient() as client_ctx:
        if client_ctx.client.collection_exists(collection_name):
            client_ctx.client.delete_collection(collection_name)


def existing_paper_ids(collection_name: str) -> set[str]:
    """Return the set of distinct paper_id payload values in a collection.

    Returns an empty set if the collection does not exist. Used by
    `--incremental` mode in the runner to skip already-indexed papers.

    Scrolls the whole collection with payload trimmed to just `paper_id`.
    For benchmark-sized collections (~500-2000 points) this is fast.
    """
    with RAGalicClient() as client_ctx:
        if not client_ctx.client.collection_exists(collection_name):
            return set()
        paper_ids: set[str] = set()
        next_page = None
        while True:
            batch, next_page = client_ctx.client.scroll(
                collection_name=collection_name,
                with_payload=["paper_id"],
                with_vectors=False,
                limit=1000,
                offset=next_page,
            )
            for point in batch:
                pid = point.payload.get("paper_id") if point.payload else None
                if pid:
                    paper_ids.add(pid)
            if next_page is None:
                break
        return paper_ids


def max_existing_node_id(collection_name: str) -> int:
    """Return the highest `id` payload integer in the collection, or -1 if empty.

    Used by `--incremental` to start the per-build node-id counter past any
    previously-indexed ids. This is critical because `_node_uuid` is keyed
    on (collection, node_id) — colliding ids would silently overwrite
    existing points.
    """
    with RAGalicClient() as client_ctx:
        if not client_ctx.client.collection_exists(collection_name):
            return -1
        max_id = -1
        next_page = None
        while True:
            batch, next_page = client_ctx.client.scroll(
                collection_name=collection_name,
                with_payload=["id"],
                with_vectors=False,
                limit=1000,
                offset=next_page,
            )
            for point in batch:
                nid = point.payload.get("id") if point.payload else None
                if isinstance(nid, int) and nid > max_id:
                    max_id = nid
            if next_page is None:
                break
        return max_id
