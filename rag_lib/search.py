from pathlib import Path
import json
from typing import Literal

from rich.console import Console
from qdrant_client.models import (
    Filter,
    FieldCondition,
    ScoredPoint,
    Prefetch,
    Document,
    MatchValue,
    MatchAny,
    FusionQuery,
    Fusion,
)
import numpy as np

from .clients import RAGalicClient
from .dataschemes import Chunk

console = Console()

BEAM_SEARCH_METHODS = Literal[
    "fixed", "adaptive_with_knee", "adaptive_with_sensitive_knee"
]

DEFAULT_COLLECTION = "ragalic"

# Default per-step fusion weights for the "weight schedule" mode.
# Step 0 = root level (lean on BM25 keywords to pick the right document);
# step N = leaf level (lean on dense embeddings to pick the right fact).
# Entries are (dense_weight, sparse_weight). The last entry is reused for any
# step beyond len(schedule)-1.
DEFAULT_FUSION_SCHEDULE: list[tuple[float, float]] = [
    (0.2, 0.8),
    (0.5, 0.5),
    (0.8, 0.2),
    (1.0, 0.0),
]


def _weights_for_step(
    schedule: list[tuple[float, float]] | None, step: int
) -> tuple[float, float] | None:
    if schedule is None:
        return None
    if not schedule:
        return None
    return schedule[min(step, len(schedule) - 1)]


def prepare_chunks(points: list[ScoredPoint]) -> list[Chunk]:
    path_to_parsed_files = Path(__file__).parent / "database" / "parsed_files"

    chunks = []
    for point in points:
        name: str = point.payload["file_name"]
        page_start = point.payload["page_start"]
        page_end = point.payload["page_end"]
        path_to_parsed_file = (path_to_parsed_files / name).with_suffix(".json")

        if path_to_parsed_file.exists():
            with open(path_to_parsed_file, "r", encoding="utf-8") as file:
                pages_content = json.load(file)
                chunk_lines = [f"FILE NAME: {name}"]
                for i in range(page_start, page_end + 1):
                    chunk_lines.append(f"<-- PAGE {i} -->")
                    chunk_lines.append(pages_content[i])
            chunk_content = "\n".join(chunk_lines)
        else:
            chunk_content = point.payload.get("text") or point.payload.get(
                "description", ""
            )

        chunk = Chunk(
            file_name=name, page_start=page_start, page_end=page_end, text=chunk_content
        )
        chunks.append(chunk)

    return chunks


def find_roots(
    query: str,
    num_to_find: int = 3,
    collection_name: str = DEFAULT_COLLECTION,
    extra_filter: Filter | None = None,
    schedule: list[tuple[float, float]] | None = None,
    step: int = 0,
) -> list[ScoredPoint]:
    """Find root nodes for the query.

    When `schedule` is None: use Qdrant's built-in unweighted RRF (unchanged
    behavior). When `schedule` is provided: use weighted RRF with the
    schedule[step] weights — lets the caller bias toward BM25 keyword match
    at the root level and dense semantics deeper in the tree.
    """
    console.print(f"Finding roots for query: {query[:20]}...", style="bold violet")
    weights = _weights_for_step(schedule, step)

    if weights is not None:
        console.print(
            f"[find_roots] weighted RRF step={step} weights={weights}",
            style="italic bright_black",
        )
        points = _weighted_rrf_query(
            collection_name=collection_name,
            query=query,
            weights=weights,
            limit=num_to_find,
            paper_filter=extra_filter,
            only_roots=True,
        )
    else:
        points = _unweighted_rrf_query(
            collection_name=collection_name,
            query=query,
            limit=num_to_find,
            paper_filter=extra_filter,
            only_roots=True,
            fetch_oversample=3,
        )
    names = [f"{point.payload['file_name']}" for point in points]
    console.print(f"Found roots of files: {names}", style="italic purple")
    return points


##################
# BRANCH SEARCH
##################


def parent_vs_children(
    query: str,
    parent: ScoredPoint,
    collection_name: str = DEFAULT_COLLECTION,
    schedule: list[tuple[float, float]] | None = None,
    step: int = 1,
) -> list[ScoredPoint]:
    file_name = parent.payload["file_name"]
    parent_id = parent.payload["id"]
    child_ids = parent.payload["child_ids"]
    all_ids = [parent_id] + child_ids

    if len(child_ids) == 0:
        console.print(
            f"Parent with id {parent_id} is leave!", style="bold underline violet"
        )
        return []

    console.print()
    console.print("#" * 20, style="red")
    console.print(
        f"Children (ids: {child_ids}) [underline]VS[/underline] Parent (id: {parent_id}) [ FILE: {file_name} ]\nFor query: {query[:20]}...",
        style="bold violet",
    )

    weights = _weights_for_step(schedule, step)
    if weights is not None:
        console.print(
            f"[parent_vs_children] weighted RRF step={step} weights={weights}",
            style="italic bright_black",
        )
        sorted_points = _weighted_rrf_query(
            collection_name=collection_name,
            query=query,
            weights=weights,
            limit=len(all_ids),
            candidate_ids=all_ids,
        )
    else:
        sorted_points = _unweighted_rrf_query(
            collection_name=collection_name,
            query=query,
            limit=len(all_ids),
            candidate_ids=all_ids,
            fetch_oversample=1,
        )

    children_better_then_parent = []
    for point in sorted_points:
        if point.payload["id"] == parent_id:
            break
        children_better_then_parent.append(point)

    ids = [child.payload["id"] for child in children_better_then_parent]
    console.print(
        f"Child ids which is better than parent: {ids if ids else 'N/A'}",
        style="italic purple",
    )
    console.print("#" * 20, style="red")
    console.print()

    return children_better_then_parent


def branch_search_points(
    query: str,
    num_roots: int = 3,
    collection_name: str = DEFAULT_COLLECTION,
    extra_root_filter: Filter | None = None,
    schedule: list[tuple[float, float]] | None = None,
) -> list[ScoredPoint]:
    roots = find_roots(
        query=query,
        num_to_find=num_roots,
        collection_name=collection_name,
        extra_filter=extra_root_filter,
        schedule=schedule,
        step=0,
    )

    final_points = []
    points_to_process = roots
    step = 1
    while len(points_to_process) > 0:
        new_point_to_process = []
        for point in points_to_process:
            new_points = parent_vs_children(
                query=query,
                parent=point,
                collection_name=collection_name,
                schedule=schedule,
                step=step,
            )
            if len(new_points) == 0:
                console.print(
                    f"--- NEW FINAL POINT (id: {point.payload['id']} | file: {point.payload['file_name']} | pages: {point.payload['page_start']} - {point.payload['page_end']}) ---",
                    style="bold green",
                )
                final_points.append(point)
            else:
                new_point_to_process.extend(new_points)
        points_to_process = new_point_to_process
        step += 1

    console.print(
        f"--- WAS FOUND {len(final_points)} POINTS FOR QUERY: '{query[:20]}...' ---",
        style="bold green on white",
    )

    return final_points


def branch_search(
    query: str,
    num_roots: int = 3,
    collection_name: str = DEFAULT_COLLECTION,
    extra_root_filter: Filter | None = None,
    schedule: list[tuple[float, float]] | None = None,
) -> list[Chunk]:
    return prepare_chunks(
        branch_search_points(
            query=query,
            num_roots=num_roots,
            collection_name=collection_name,
            extra_root_filter=extra_root_filter,
            schedule=schedule,
        )
    )


##################
# BEAM SEARCH
##################


def check_ids(old_ids: list, new_ids: list):
    return set(old_ids) == set(new_ids)


def cut_knee(points: list[ScoredPoint]) -> list[ScoredPoint]:
    """
    Отсекает 'хвост' результатов, находя точку максимального изгиба (колено)
    на графике RRF-скоров. Нормированная версия, которая учитывает разные масштабы по осям.
    """
    n_points = len(points)

    if n_points <= 2:
        return points

    scores = np.array([p.score for p in points])
    x = np.arange(n_points)
    y = scores

    x_norm = (x - x.min()) / (x.max() - x.min())

    y_range = y.max() - y.min()
    if y_range == 0:
        return points
    y_norm = (y - y.min()) / y_range

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    points_vec = np.vstack([x_norm - p1[0], y_norm - p1[1]]).T

    scalar_product = (
        points_vec[:, 0] * line_vec_norm[1] - points_vec[:, 1] * line_vec_norm[0]
    )
    distances = np.abs(scalar_product)

    knee_idx = np.argmax(distances)

    return points[: knee_idx + 1]


def cut_knee_flexible(
    points: list[ScoredPoint], sensitivity: float = 0.5
) -> list[ScoredPoint]:
    """
    Отсекает 'хвост' результатов, находя точку максимального изгиба (колено)
    на графике RRF-скоров. Нормированная версия, которая учитывает разные масштабы по осям и позволяет более гибко настраивать чувствительность среза.
    """
    n_points = len(points)

    if n_points <= 2:
        return points

    scores = np.array([p.score for p in points])
    x = np.arange(n_points)
    y = scores

    x_norm = (x - x.min()) / (x.max() - x.min())

    y_range = y.max() - y.min()
    if y_range == 0:
        return points
    y_norm = (y - y.min()) / y_range

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    points_vec = np.vstack([x_norm - p1[0], y_norm - p1[1]]).T

    scalar_product = (
        points_vec[:, 0] * line_vec_norm[1] - points_vec[:, 1] * line_vec_norm[0]
    )
    distances = np.abs(scalar_product)

    knee_idx = np.argmax(distances)

    max_dist = np.max(distances)

    threshold = max_dist * sensitivity

    indices_above_threshold = np.where(distances >= threshold)[0]
    knee_idx = indices_above_threshold[-1]

    return points[: knee_idx + 1]


def parents_vs_children(
    query: str,
    parents: list[ScoredPoint],
    width: int = 3,
    search_method: BEAM_SEARCH_METHODS = "fixed",
    sensitivity: float = 0.85,
    collection_name: str = DEFAULT_COLLECTION,
    schedule: list[tuple[float, float]] | None = None,
    step: int = 1,
) -> list[ScoredPoint]:
    task_meta = {
        "file_names": [],
        "parent_ids": [],
        "child_ids": [],
    }

    for point in parents:
        task_meta["file_names"].append(point.payload["file_name"])
        task_meta["parent_ids"].append(point.payload["id"])
        task_meta["child_ids"].extend(point.payload["child_ids"])

    task_meta["file_names"] = list(set(task_meta["file_names"]))
    task_meta["parent_ids"] = list(set(task_meta["parent_ids"]))
    task_meta["child_ids"] = list(set(task_meta["child_ids"]))

    if len(task_meta["child_ids"]) == 0:
        console.print(
            f"Parents with ids ({task_meta['parent_ids']}) is all leaves!",
            style="bold underline violet",
        )
        return parents

    console.print("")
    console.print("#" * 20, style="red")
    console.print(
        f"Running [underline]ALL VS ALL[/underline] for:\nParent IDs: {task_meta['parent_ids']} | Files: {task_meta['file_names']}",
        style="bold violet",
    )
    all_ids = task_meta["parent_ids"] + task_meta["child_ids"]

    weights = _weights_for_step(schedule, step)
    if weights is not None:
        console.print(
            f"[parents_vs_children] weighted RRF step={step} weights={weights}",
            style="italic bright_black",
        )
        sorted_points = _weighted_rrf_query(
            collection_name=collection_name,
            query=query,
            weights=weights,
            limit=len(all_ids),
            candidate_ids=all_ids,
        )
    else:
        sorted_points = _unweighted_rrf_query(
            collection_name=collection_name,
            query=query,
            limit=len(all_ids),
            candidate_ids=all_ids,
            fetch_oversample=1,
        )

    children_to_eliminate = []
    parents_to_eliminate = []
    new_top_k = []

    if search_method == "adaptive_with_knee":
        console.print(
            f"Applying KNEE method to cut the tail of results. Initial candidates: {len(sorted_points)}",
            style="italic bright_black",
        )
        sorted_points = cut_knee(sorted_points)
        console.print(
            f"Candidates after KNEE cut: {len(sorted_points)}",
            style="italic bright_black",
        )
    elif search_method == "adaptive_with_sensitive_knee":
        console.print(
            f"Applying SENSITIVE KNEE method to cut the tail of results with sensitivity {sensitivity}. Initial candidates: {len(sorted_points)}",
            style="italic bright_black",
        )
        sorted_points = cut_knee_flexible(sorted_points, sensitivity=sensitivity)
        console.print(
            f"Candidates after SENSITIVE KNEE cut: {len(sorted_points)}",
            style="italic bright_black",
        )

    for point in sorted_points:
        p_id = point.payload["id"]

        # Логика исключения (Suppression)
        if p_id in task_meta["child_ids"] and p_id not in children_to_eliminate:
            # Если ребенок лучше родителя — родителю тут не место
            parents_to_eliminate.append(point.payload["parent_id"])
            new_top_k.append(point)

        elif p_id in task_meta["parent_ids"] and p_id not in parents_to_eliminate:
            # Если родитель лучше детей — убираем его детей из пула
            children_to_eliminate.extend(point.payload["child_ids"])
            new_top_k.append(point)

        if search_method == "fixed" and len(new_top_k) >= width:
            break

    new_top_k_meta = [
        f"(ID: {point.payload['id']} | FILE: {point.payload['file_name']} | PAGES: {point.payload['page_start']} - {point.payload['page_end']})"
        for point in new_top_k
    ]
    console.print(
        f"IDs: {task_meta['parent_ids']} | Files: {task_meta['file_names']} [OLD TOP]",
        style="italic purple",
    )
    console.print("|\nY", style="bold red")
    console.print(f"{', '.join(new_top_k_meta)} [NEW TOP]", style="italic purple")
    console.print("#" * 20, style="red")
    console.print("")

    return new_top_k


def beam_search_points(
    query: str,
    beam_width: int = 3,
    search_method: BEAM_SEARCH_METHODS = "fixed",
    max_num_roots: int = 20,
    sensitivity: float = 0.5,
    collection_name: str = DEFAULT_COLLECTION,
    extra_root_filter: Filter | None = None,
    schedule: list[tuple[float, float]] | None = None,
) -> list[ScoredPoint]:
    if search_method == "fixed":
        console.print(
            f"Running BEAM SEARCH with [underline]FIXED[/underline] width: {beam_width}",
            style="bold cyan",
        )
        old_points = find_roots(
            query=query,
            num_to_find=beam_width,
            collection_name=collection_name,
            extra_filter=extra_root_filter,
            schedule=schedule,
            step=0,
        )
    elif (
        search_method == "adaptive_with_knee"
        or search_method == "adaptive_with_sensitive_knee"
    ):
        if search_method == "adaptive_with_sensitive_knee":
            console.print(
                f"Running BEAM SEARCH with [underline]ADAPTIVE[/underline] width using [underline]SENSITIVE KNEE[/underline] method with sensitivity {sensitivity}",
                style="bold cyan",
            )
        else:
            console.print(
                f"Running BEAM SEARCH with [underline]ADAPTIVE[/underline] width using [underline]KNEE[/underline] method",
                style="bold cyan",
            )
        with RAGalicClient() as client:
            must = [FieldCondition(key="parent_id", match=MatchValue(value=-1))]
            if extra_root_filter is not None and extra_root_filter.must:
                must.extend(extra_root_filter.must)  # type: ignore
            root_filter = Filter(must=must)
            all_root_points_num = client.client.count(
                collection_name=collection_name, count_filter=root_filter
            ).count
            all_root_points_num = min(all_root_points_num, max_num_roots)
        console.print(
            f"Total root points available: {all_root_points_num}",
            style="italic bright_black",
        )
        old_points = find_roots(
            query=query,
            num_to_find=all_root_points_num,
            collection_name=collection_name,
            extra_filter=extra_root_filter,
            schedule=schedule,
            step=0,
        )
        console.print(
            f"Initial root points retrieved: {len(old_points)}",
            style="italic bright_black",
        )
        if search_method == "adaptive_with_knee":
            old_points = cut_knee(old_points)
            console.print(
                f"Root points after KNEE cut: {len(old_points)}",
                style="italic bright_black",
            )
        elif search_method == "adaptive_with_sensitive_knee":
            old_points = cut_knee_flexible(old_points, sensitivity=sensitivity)
            console.print(
                f"Root points after KNEE cut: {len(old_points)}",
                style="italic bright_black",
            )
    new_points = []

    old_ids = [point.payload["id"] for point in old_points]
    new_ids = []
    step = 1
    while not check_ids(old_ids=old_ids, new_ids=new_ids):
        old_ids = [point.payload["id"] for point in old_points]
        new_points = parents_vs_children(
            query=query,
            parents=old_points,
            width=beam_width,
            search_method=search_method,
            sensitivity=sensitivity,
            collection_name=collection_name,
            schedule=schedule,
            step=step,
        )
        old_points = new_points
        new_ids = [point.payload["id"] for point in new_points]
        step += 1

    console.print(
        f"--- WAS FOUND {len(new_points)} POINTS FOR QUERY: '{query[:20]}...' ---",
        style="bold green on white",
    )

    return new_points


def beam_search(
    query: str,
    beam_width: int = 3,
    search_method: BEAM_SEARCH_METHODS = "fixed",
    max_num_roots: int = 20,
    sensitivity: float = 0.5,
    collection_name: str = DEFAULT_COLLECTION,
    extra_root_filter: Filter | None = None,
    schedule: list[tuple[float, float]] | None = None,
) -> list[Chunk]:
    return prepare_chunks(
        beam_search_points(
            query=query,
            beam_width=beam_width,
            search_method=search_method,
            max_num_roots=max_num_roots,
            sensitivity=sensitivity,
            collection_name=collection_name,
            extra_root_filter=extra_root_filter,
            schedule=schedule,
        )
    )


##################
# SCHEDULED-FUSION BEAM SEARCH
# ──────────────────────────────────────────────────────────────────────────
# Hypothesis: at the document/section level (root, near-root), keyword
# matching (BM25) is the right signal for "which topic is this?"; at the
# leaf level, dense embeddings are the right signal for "which fact in
# this topic?". We expose a per-step weight schedule between dense and
# sparse, fused with weighted RRF in Python (Qdrant's built-in fusion has
# no weight knob).
##################


def _build_must_clauses(
    *,
    paper_filter: Filter | None,
    candidate_ids: list[int] | None,
    only_roots: bool,
) -> list:
    """Shared filter-clause assembly used by both the unweighted and
    weighted RRF helpers — keeps the must-list construction in one place."""
    must: list = []
    if only_roots:
        must.append(FieldCondition(key="parent_id", match=MatchValue(value=-1)))
    if paper_filter is not None and paper_filter.must:
        must.extend(paper_filter.must)  # type: ignore[arg-type]
    if candidate_ids is not None:
        must.append(FieldCondition(key="id", match=MatchAny(any=candidate_ids)))
    return must


def _unweighted_rrf_query(
    *,
    collection_name: str,
    query: str,
    limit: int,
    candidate_ids: list[int] | None = None,
    paper_filter: Filter | None = None,
    only_roots: bool = False,
    fetch_oversample: int = 3,
) -> list[ScoredPoint]:
    """One Qdrant query using built-in unweighted RRF fusion of dense+sparse.

    Mirrors the surface of `_weighted_rrf_query` (minus the weights arg) so
    callers can flip between the two without rebuilding their own filter
    plumbing. Behavior is identical to the previous inline `query_points(
    prefetch=[dense, sparse], query=FusionQuery(Fusion.RRF))` pattern.
    """
    fetch_limit = max(limit * fetch_oversample, limit)
    must = _build_must_clauses(
        paper_filter=paper_filter,
        candidate_ids=candidate_ids,
        only_roots=only_roots,
    )
    f: Filter | None = Filter(must=must) if must else None

    with RAGalicClient() as client_ctx:
        client = client_ctx.client
        dense_model: str = client.embedding_model_name  # type: ignore
        sparse_model: str = client.sparse_embedding_model_name  # type: ignore
        return client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=dense_model),
                    using="dense",
                    limit=fetch_limit,
                    filter=f,
                ),
                Prefetch(
                    query=Document(text=query, model=sparse_model),
                    using="sparse",
                    limit=fetch_limit,
                    filter=f,
                ),
            ],
            query_filter=f,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        ).points


def _weighted_rrf_query(
    *,
    collection_name: str,
    query: str,
    weights: tuple[float, float],
    limit: int,
    candidate_ids: list[int] | None = None,
    paper_filter: Filter | None = None,
    only_roots: bool = False,
    fetch_oversample: int = 3,
    rrf_k: int = 60,
) -> list[ScoredPoint]:
    """One query against the dense and sparse vectors, fused with weighted RRF.

    weights is (dense_weight, sparse_weight). Either may be 0 to skip that
    side entirely. Returns up to `limit` ScoredPoints sorted by the combined
    score, with `.score` overwritten to the weighted-RRF value.
    """
    dense_w, sparse_w = weights
    fetch_limit = max(limit * fetch_oversample, limit)

    must = _build_must_clauses(
        paper_filter=paper_filter,
        candidate_ids=candidate_ids,
        only_roots=only_roots,
    )
    f: Filter | None = Filter(must=must) if must else None

    with RAGalicClient() as client_ctx:
        client = client_ctx.client
        dense_model: str = client.embedding_model_name  # type: ignore
        sparse_model: str = client.sparse_embedding_model_name  # type: ignore

        dense_points: list[ScoredPoint] = []
        if dense_w > 0:
            dense_points = client.query_points(
                collection_name=collection_name,
                query=Document(text=query, model=dense_model),
                using="dense",
                query_filter=f,
                limit=fetch_limit,
            ).points

        sparse_points: list[ScoredPoint] = []
        if sparse_w > 0:
            sparse_points = client.query_points(
                collection_name=collection_name,
                query=Document(text=query, model=sparse_model),
                using="sparse",
                query_filter=f,
                limit=fetch_limit,
            ).points

    id_to_point: dict[int, ScoredPoint] = {}
    scores: dict[int, float] = {}
    for rank, p in enumerate(dense_points, 1):
        pid = p.payload["id"]
        id_to_point[pid] = p
        scores[pid] = scores.get(pid, 0.0) + dense_w / (rrf_k + rank)
    for rank, p in enumerate(sparse_points, 1):
        pid = p.payload["id"]
        if pid not in id_to_point:
            id_to_point[pid] = p
        scores[pid] = scores.get(pid, 0.0) + sparse_w / (rrf_k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda i: -scores[i])
    out: list[ScoredPoint] = []
    for pid in sorted_ids[:limit]:
        p = id_to_point[pid]
        try:
            p.score = scores[pid]  # mutate so downstream sees the fused score
        except Exception:
            pass
        out.append(p)
    return out


def beam_search_points_scheduled(
    query: str,
    schedule: list[tuple[float, float]],
    beam_width: int = 3,
    collection_name: str = DEFAULT_COLLECTION,
    extra_root_filter: Filter | None = None,
    max_iter: int = 8,
) -> list[ScoredPoint]:
    """Fixed-width beam search with per-step (dense, sparse) RRF weights.

    `schedule[i]` is the weight pair used at iteration i. The last entry is
    re-used for any iteration beyond `len(schedule) - 1`. The expansion logic
    (parent-vs-child suppression) is identical to `beam_search_points`; only
    the candidate scoring differs.
    """
    console.print(
        f"Running BEAM SEARCH with [underline]SCHEDULED FUSION[/underline] "
        f"width={beam_width} schedule={schedule}",
        style="bold cyan",
    )

    weights_root = schedule[0]
    old_points = _weighted_rrf_query(
        collection_name=collection_name,
        query=query,
        weights=weights_root,
        limit=beam_width,
        paper_filter=extra_root_filter,
        only_roots=True,
    )

    old_ids: list = [p.payload["id"] for p in old_points]
    new_ids: list = []
    step = 1
    new_points: list[ScoredPoint] = list(old_points)

    while not check_ids(old_ids=old_ids, new_ids=new_ids) and step <= max_iter:
        weights = schedule[min(step, len(schedule) - 1)]
        old_ids = [p.payload["id"] for p in old_points]

        parent_ids = list({p.payload["id"] for p in old_points})
        child_ids_set: set[int] = set()
        for p in old_points:
            for cid in p.payload.get("child_ids", []) or []:
                child_ids_set.add(cid)
        child_ids = list(child_ids_set)

        if not child_ids:
            console.print(
                f"[scheduled] step {step}: all current beam members are leaves",
                style="italic bright_black",
            )
            new_points = list(old_points)
            break

        all_candidate_ids = parent_ids + child_ids
        console.print(
            f"[scheduled] step {step}: weights={weights} "
            f"parents={len(parent_ids)} children={len(child_ids)}",
            style="italic bright_black",
        )

        sorted_points = _weighted_rrf_query(
            collection_name=collection_name,
            query=query,
            weights=weights,
            limit=len(all_candidate_ids),
            candidate_ids=all_candidate_ids,
        )

        children_to_eliminate: list[int] = []
        parents_to_eliminate: list = []
        new_top_k: list[ScoredPoint] = []
        for point in sorted_points:
            p_id = point.payload["id"]
            if p_id in child_ids and p_id not in children_to_eliminate:
                parents_to_eliminate.append(point.payload.get("parent_id"))
                new_top_k.append(point)
            elif p_id in parent_ids and p_id not in parents_to_eliminate:
                children_to_eliminate.extend(
                    point.payload.get("child_ids", []) or []
                )
                new_top_k.append(point)
            if len(new_top_k) >= beam_width:
                break

        old_points = new_top_k
        new_points = new_top_k
        new_ids = [p.payload["id"] for p in new_points]
        step += 1

    console.print(
        f"--- SCHEDULED BEAM SEARCH found {len(new_points)} POINTS "
        f"after {step - 1} step(s) for query: '{query[:20]}...' ---",
        style="bold green on white",
    )
    return new_points


if __name__ == "__main__":
    test_query = "Which laws are administered by the Registrar and what are their respective citation titles?"
    branch_search(query=test_query)
    beam_search(query=test_query)
    beam_search(query=test_query, search_method="adaptive_with_knee")
    beam_search(
        query=test_query, search_method="adaptive_with_sensitive_knee", sensitivity=0.9
    )
