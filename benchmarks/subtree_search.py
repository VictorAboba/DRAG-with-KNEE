"""DRAG-Subtree: retrieval that returns coherent document subtrees, not chunks.

The standard retrieval interface is ``query -> top-K independent passages``.
Downstream LLM generators then have to reconstruct narrative coherence from
disjoint fragments.

DRAG-Subtree changes the unit of retrieval:

    query -> ranked list of *subtrees*

A subtree is (parent_node, kept_children_subset). The knee on the
parent-vs-children score distribution decides the *scope* of the answer:

    parent score on top              -> theme is broad, return parent + cut_knee(children)
    one child dominates              -> answer is local, drill into that child recursively
    children flat & comparable       -> answer distributed, return parent + all surviving children

This module is benchmark-side: it reads from an existing bench_drag
collection, no re-indexing.

Two stages:

1. **Content-aware paper selection.** A top-N RRF search over the whole
   tree (NOT restricted to roots) collects nodes that match the query.
   The unique papers represented by those nodes become the candidate set.
   This fixes the 47% wrong-paper rate we measured on the standard DRAG
   variants where ``find_roots`` (only_roots=True) selects papers via
   their root-summary embedding alone.

2. **Recursive descent per paper.** Each candidate paper's tree is walked
   from root downward, applying the three-case rule above. Scores are
   pre-fetched once per paper (one RRF query over all nodes in the paper)
   so the descent is in-memory after that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_lib.clients import RAGalicClient
from rag_lib.search import _unweighted_rrf_query, cut_knee_flexible


# How dominant the leading child must be to trigger drill-down instead of
# returning parent+children as a subtree. Score ratio.
DOMINATE_RATIO = 1.3

# How dominant the parent must be to short-circuit and stop descent.
PARENT_WIN_MARGIN = 1.1

# Sensitivity used for the knee that prunes children inside a returned subtree.
# Lower = keep more children. We default to 0.5 — mid-range.
KNEE_SENSITIVITY = 0.5

# How many seed nodes (full-tree top-K) we pull in stage 1 before deduping
# to unique papers. Wider = better recall of which paper has the answer.
SEED_FETCH = 30


@dataclass
class Subtree:
    """One returned context unit.

    ``root_node_id`` is the highest node we kept (parent of the returned
    children, or a single leaf if drill-down went all the way down).
    ``kept_node_ids`` are the specific nodes returned (including the root).
    ``descendant_leaf_paragraph_ids`` flattens kept_node_ids to leaf-level
    paragraph_ids in document order — what the metric module uses to score
    against gold evidence.
    ``score`` is the parent's RRF score (or the leaf's if it's a leaf).
    """

    root_node_id: int
    paper_id: str
    kept_node_ids: list[int]
    descendant_leaf_paragraph_ids: list[str]
    score: float
    decision: str  # "leaf", "parent_dominates", "drill_down", "knee_split"


def _scroll_paper_nodes(
    collection_name: str, paper_id: str
) -> dict[int, dict]:
    """Fetch every node payload for one paper. Returns a dict keyed by
    the integer node-id (payload['id']).

    The tree size per paper is small (tens to a few hundred nodes), so a
    single scroll is cheap. We need ``child_ids`` and ``parent_id`` to walk
    the tree without further round-trips.
    """
    paper_filter = Filter(
        must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
    )
    by_id: dict[int, dict] = {}
    next_offset: Optional[object] = None
    with RAGalicClient() as ctx:
        client = ctx.client
        while True:
            page, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=paper_filter,
                with_payload=True,
                with_vectors=False,
                limit=512,
                offset=next_offset,
            )
            for pt in page:
                if pt.payload:
                    by_id[pt.payload["id"]] = pt.payload
            if next_offset is None:
                break
    return by_id


def _score_paper_nodes(
    collection_name: str, paper_id: str, query: str, n_nodes: int
) -> dict[int, float]:
    """Single RRF query over EVERY node in the paper. Returns score-by-id.

    Nodes the RRF didn't rank get a default score of 0.0 at the call site
    via ``dict.get(id, 0.0)``.
    """
    paper_filter = Filter(
        must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
    )
    scored = _unweighted_rrf_query(
        collection_name=collection_name,
        query=query,
        limit=max(n_nodes, 50),
        paper_filter=paper_filter,
        only_roots=False,
    )
    return {p.payload["id"]: p.score for p in scored}


def _candidate_papers(
    collection_name: str,
    query: str,
    paper_filter: Filter | None,
    max_papers: int,
) -> list[tuple[str, float]]:
    """Stage 1: find top papers via leaf-level content match.

    Restricts the seed RRF to leaves (``is_leaf=True``) so parent and root
    summary embeddings — which can vote a paper into the top set even
    when its actual content doesn't match — don't dominate. Dedupes the
    leaf hits to unique paper_ids in score order and returns the top
    ``max_papers``.

    Mirrors what ``hybrid_rrf_flat`` retrieves on bench_flat: pure leaf
    content matching, then group-by-paper. The DRAG-Subtree win, if there
    is one, is in the *recursive descent* picking the right scope within
    the right paper — not in re-inventing paper selection.
    """
    leaf_clause = FieldCondition(key="is_leaf", match=MatchValue(value=True))
    must = [leaf_clause]
    if paper_filter is not None and paper_filter.must:
        must.extend(paper_filter.must)  # type: ignore[arg-type]
    leaf_only_filter = Filter(must=must)
    seeds = _unweighted_rrf_query(
        collection_name=collection_name,
        query=query,
        limit=SEED_FETCH,
        paper_filter=leaf_only_filter,
        only_roots=False,
    )
    seen: dict[str, float] = {}
    for p in seeds:
        pid = p.payload.get("paper_id")
        if not pid or pid in seen:
            continue
        seen[pid] = p.score
        if len(seen) >= max_papers:
            break
    return sorted(seen.items(), key=lambda kv: -kv[1])


def _descendant_leaves(
    by_id: dict[int, dict], root_node_id: int
) -> list[str]:
    """Flatten a node's subtree down to leaf paragraph_ids in document
    order (sorted by page_start as a stable proxy)."""
    out: list[dict] = []

    def walk(nid: int) -> None:
        node = by_id.get(nid)
        if node is None:
            return
        if node.get("is_leaf"):
            out.append(node)
            return
        for cid in node.get("child_ids", []) or []:
            walk(cid)

    walk(root_node_id)
    out.sort(key=lambda n: n.get("page_start", 0))
    return [n["paragraph_id"] for n in out if n.get("paragraph_id")]


def _descend(
    by_id: dict[int, dict],
    scores: dict[int, float],
    node: dict,
    paper_id: str,
    knee_sensitivity: float,
    parent_win_margin: float = PARENT_WIN_MARGIN,
    dominate_ratio: float = DOMINATE_RATIO,
) -> Subtree:
    """Recursive descent with the three-case knee rule.

    Returns one Subtree representing where descent stopped:

      - leaf reached            -> single-leaf subtree
      - parent_dominates        -> parent on top of [parent + children] -> return parent's whole subtree
                                                                          (with optional pruning via knee on children)
      - drill_down              -> one child dominates the rest -> recurse into it
      - knee_split              -> children comparable -> keep top-knee children, return as subtree under parent
    """
    nid = node["id"]
    parent_score = scores.get(nid, 0.0)

    if node.get("is_leaf"):
        return Subtree(
            root_node_id=nid,
            paper_id=paper_id,
            kept_node_ids=[nid],
            descendant_leaf_paragraph_ids=[node["paragraph_id"]] if node.get("paragraph_id") else [],
            score=parent_score,
            decision="leaf",
        )

    child_ids: list[int] = node.get("child_ids", []) or []
    if not child_ids:
        # Defensive: non-leaf node with no children — treat as a single point.
        return Subtree(
            root_node_id=nid,
            paper_id=paper_id,
            kept_node_ids=[nid],
            descendant_leaf_paragraph_ids=_descendant_leaves(by_id, nid),
            score=parent_score,
            decision="leaf",
        )

    # Sort children by their RRF score. Children unranked by the global
    # query get score 0.0.
    children_with_scores: list[tuple[int, float]] = sorted(
        ((cid, scores.get(cid, 0.0)) for cid in child_ids),
        key=lambda cs: -cs[1],
    )

    top_child_score = children_with_scores[0][1]

    # Case 1: parent beats every child — the parent's abstract captures the
    # theme better than any single child. Return this subtree as a unit.
    if parent_score > top_child_score * parent_win_margin:
        return Subtree(
            root_node_id=nid,
            paper_id=paper_id,
            kept_node_ids=[nid] + [cid for cid, _ in children_with_scores],
            descendant_leaf_paragraph_ids=_descendant_leaves(by_id, nid),
            score=parent_score,
            decision="parent_dominates",
        )

    # Case 2: one child dominates the rest — drill down recursively.
    if len(children_with_scores) >= 2:
        if top_child_score > children_with_scores[1][1] * dominate_ratio:
            top_child_id = children_with_scores[0][0]
            return _descend(
                by_id, scores, by_id[top_child_id], paper_id, knee_sensitivity,
                parent_win_margin=parent_win_margin,
                dominate_ratio=dominate_ratio,
            )

    # Case 3: children comparable. Apply a knee to decide how many to keep.
    # We construct a fake ScoredPoint-shaped list for cut_knee_flexible:
    # it only reads .score, so a lightweight stand-in works.
    class _SP:
        __slots__ = ("score",)

        def __init__(self, s: float) -> None:
            self.score = s

    knee_input = [_SP(s) for _, s in children_with_scores]
    survivors_count = len(cut_knee_flexible(knee_input, sensitivity=knee_sensitivity))
    kept_pairs = children_with_scores[:max(1, survivors_count)]

    if len(kept_pairs) == 1:
        # Single survivor → drill down into it
        only_child_id = kept_pairs[0][0]
        return _descend(
            by_id, scores, by_id[only_child_id], paper_id, knee_sensitivity,
            parent_win_margin=parent_win_margin,
            dominate_ratio=dominate_ratio,
        )

    kept_child_ids = [cid for cid, _ in kept_pairs]
    # Subtree = parent + the surviving children (each child's whole subtree
    # is implicitly included for leaf flattening, since we walk by_id).
    descendant_leaves: list[str] = []
    seen_leaves: set[str] = set()
    for cid in kept_child_ids:
        for pid in _descendant_leaves(by_id, cid):
            if pid not in seen_leaves:
                seen_leaves.add(pid)
                descendant_leaves.append(pid)

    return Subtree(
        root_node_id=nid,
        paper_id=paper_id,
        kept_node_ids=[nid] + kept_child_ids,
        descendant_leaf_paragraph_ids=descendant_leaves,
        score=parent_score,
        decision="knee_split",
    )


def subtree_search(
    query: str,
    paper_id: str | None,
    collection_name: str,
    max_papers: int = 3,
    knee_sensitivity: float = KNEE_SENSITIVITY,
    parent_win_margin: float = PARENT_WIN_MARGIN,
    dominate_ratio: float = DOMINATE_RATIO,
) -> list[Subtree]:
    """Public entry point. Returns a ranked list of Subtree objects.

    ``paper_id=None`` (cross-paper): stage-1 selects top ``max_papers`` papers
    via content match. ``paper_id="..."`` (per-paper): skip stage 1, descend
    once in the named paper.
    """
    paper_filter = (
        Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])
        if paper_id is not None
        else None
    )

    if paper_id is not None:
        candidates: list[tuple[str, float]] = [(paper_id, 1.0)]
    else:
        candidates = _candidate_papers(
            collection_name=collection_name,
            query=query,
            paper_filter=None,
            max_papers=max_papers,
        )
    if not candidates:
        return []

    subtrees: list[Subtree] = []
    for pid, _seed_score in candidates:
        by_id = _scroll_paper_nodes(collection_name, pid)
        if not by_id:
            continue
        root = next(
            (n for n in by_id.values() if n.get("parent_id") == -1),
            None,
        )
        if root is None:
            continue
        scores = _score_paper_nodes(collection_name, pid, query, n_nodes=len(by_id))
        st = _descend(
            by_id, scores, root, pid, knee_sensitivity,
            parent_win_margin=parent_win_margin,
            dominate_ratio=dominate_ratio,
        )
        subtrees.append(st)

    subtrees.sort(key=lambda s: -s.score)
    return subtrees
