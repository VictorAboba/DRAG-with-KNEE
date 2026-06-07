"""QASPER loader + gold-evidence mapping.

QASPER (Dasigi et al., 2021) provides scientific papers with question-answer pairs
where each answer cites *evidence paragraphs* from the paper. This gives us
ground-truth chunk IDs for retrieval-only evaluation.

Schema returned by `load_qasper_slice`:

    Paper:
        paper_id: str
        title: str
        paragraphs: list[Paragraph]   # all section paragraphs, in document order
    Paragraph:
        paragraph_id: str             # f"{paper_id}__s{section_idx}_p{para_idx}"
        section_idx: int
        section_name: str
        para_idx: int                 # index within section
        text: str
        global_idx: int               # index within the whole paper (for tree building)
    Question:
        question_id: str
        paper_id: str
        text: str
        gold_paragraph_ids: list[str] # union over all annotators
        is_unanswerable: bool
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class Paragraph:
    paragraph_id: str
    section_idx: int
    section_name: str
    para_idx: int
    text: str
    global_idx: int


@dataclass
class Paper:
    paper_id: str
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "paragraphs": [asdict(p) for p in self.paragraphs],
        }


@dataclass
class Question:
    question_id: str
    paper_id: str
    text: str
    gold_paragraph_ids: list[str]
    is_unanswerable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    """Cheap whitespace normalization for evidence-paragraph matching."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _match_evidence_to_paragraphs(
    evidence: str, paragraphs: list[Paragraph]
) -> list[str]:
    """Map an evidence string to one or more paragraph_ids.

    QASPER evidence strings are typically exact paragraph copies (sometimes
    truncated). We accept exact-equal-after-normalization OR substring-contains
    in either direction.
    """
    if not evidence:
        return []
    norm_ev = _normalize(evidence)
    if not norm_ev:
        return []

    hits: list[str] = []
    for p in paragraphs:
        norm_p = _normalize(p.text)
        if not norm_p:
            continue
        if norm_p == norm_ev or norm_ev in norm_p or norm_p in norm_ev:
            hits.append(p.paragraph_id)
    return hits


def _flatten_full_text(paper_raw: dict, paper_id: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    global_idx = 1  # 1-based to align with rag_lib Node.page_start >= 1
    full_text = paper_raw.get("full_text") or {}

    section_names: list[str] = full_text.get("section_name") or []
    paragraphs_lists: list[list[str]] = full_text.get("paragraphs") or []

    for si, (sec_name, sec_paragraphs) in enumerate(
        zip(section_names, paragraphs_lists)
    ):
        for pi, para in enumerate(sec_paragraphs):
            text = (para or "").strip()
            if not text:
                continue
            paragraphs.append(
                Paragraph(
                    paragraph_id=f"{paper_id}__s{si}_p{pi}",
                    section_idx=si,
                    section_name=sec_name or f"section_{si}",
                    para_idx=pi,
                    text=text,
                    global_idx=global_idx,
                )
            )
            global_idx += 1
    return paragraphs


def _extract_evidence_strings(ans_entry: dict) -> tuple[list[str], bool]:
    """Pull evidence strings and the 'unanswerable' flag from a QASPER question.

    QASPER HF schema for one question's annotations:
        ans_entry = {
            'answer':         list[dict],   # one entry per annotator
            'annotation_id':  list[str],
            'worker_id':      list[str],
        }
    Each annotator's `answer` dict has `evidence` and `highlighted_evidence`
    fields (both `list[str]`).
    """
    if not isinstance(ans_entry, dict):
        return [], False
    answers: list[dict] = ans_entry.get("answer") or []
    evidence_strings: list[str] = []
    unanswerable = False
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        if ans.get("unanswerable"):
            unanswerable = True
        for ev in ans.get("evidence") or []:
            if isinstance(ev, str) and ev.strip():
                evidence_strings.append(ev)
        for ev in ans.get("highlighted_evidence") or []:
            if isinstance(ev, str) and ev.strip():
                evidence_strings.append(ev)
    return evidence_strings, unanswerable


def _stable_paper_id(paper_idx: int, paper_raw: dict) -> str:
    raw_id = paper_raw.get("id") or paper_raw.get("paper_id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    title = paper_raw.get("title") or ""
    h = hashlib.md5(title.encode("utf-8", "ignore")).hexdigest()[:8]
    return f"paper{paper_idx:03d}_{h}"


def load_qasper_slice(
    num_papers: int = 5,
    max_questions: int = 50,
    split: str = "validation",
    cache_dir: str | None = None,
    seed: int = 0,
) -> tuple[list[Paper], list[Question]]:
    """Download (or load from cache) a slice of QASPER.

    Returns (papers, questions). Questions are filtered to those with at least
    one matchable gold paragraph and dropped if marked unanswerable by every
    annotator. Slice is reproducible: papers are taken in sorted-by-id order
    after a seeded shuffle.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("allenai/qasper", split=split, cache_dir=cache_dir)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    papers: list[Paper] = []
    questions: list[Question] = []

    for paper_idx in indices:
        if len(papers) >= num_papers and len(questions) >= max_questions:
            break

        paper_raw = ds[paper_idx]
        paper_id = _stable_paper_id(paper_idx, paper_raw)
        flat_paragraphs = _flatten_full_text(paper_raw, paper_id)
        if len(flat_paragraphs) < 3:
            continue

        qas_lists = paper_raw.get("qas") or {}
        questions_for_paper: list[Question] = []

        question_texts = qas_lists.get("question") or []
        question_ids = qas_lists.get("question_id") or []
        answers_list = qas_lists.get("answers") or []

        for qi, qtext in enumerate(question_texts):
            qid = (
                question_ids[qi]
                if qi < len(question_ids)
                else f"{paper_id}_q{qi}"
            )
            ans_entry = answers_list[qi] if qi < len(answers_list) else {}
            evidence, unanswerable = _extract_evidence_strings(ans_entry)

            gold_ids: list[str] = []
            for ev in evidence:
                gold_ids.extend(_match_evidence_to_paragraphs(ev, flat_paragraphs))
            gold_ids = sorted(set(gold_ids))

            if unanswerable and not gold_ids:
                continue
            if not gold_ids:
                continue

            questions_for_paper.append(
                Question(
                    question_id=str(qid),
                    paper_id=paper_id,
                    text=qtext,
                    gold_paragraph_ids=gold_ids,
                    is_unanswerable=unanswerable,
                )
            )

        if not questions_for_paper:
            continue

        papers.append(
            Paper(
                paper_id=paper_id,
                title=paper_raw.get("title") or paper_id,
                paragraphs=flat_paragraphs,
            )
        )
        questions.extend(questions_for_paper)
        if len(questions) >= max_questions:
            questions = questions[:max_questions]
            break

    if len(papers) > num_papers:
        keep_paper_ids = {p.paper_id for p in papers[:num_papers]}
        papers = [p for p in papers if p.paper_id in keep_paper_ids]
        questions = [q for q in questions if q.paper_id in keep_paper_ids][:max_questions]

    return papers, questions


def save_slice(
    papers: Iterable[Paper], questions: Iterable[Question], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "papers.json", "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in papers], f, ensure_ascii=False, indent=2)
    with open(out_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in questions], f, ensure_ascii=False, indent=2)


def load_slice(in_dir: Path) -> tuple[list[Paper], list[Question]]:
    with open(in_dir / "papers.json", "r", encoding="utf-8") as f:
        papers_raw = json.load(f)
    with open(in_dir / "questions.json", "r", encoding="utf-8") as f:
        questions_raw = json.load(f)

    papers: list[Paper] = []
    for p in papers_raw:
        papers.append(
            Paper(
                paper_id=p["paper_id"],
                title=p["title"],
                paragraphs=[Paragraph(**para) for para in p["paragraphs"]],
            )
        )
    questions = [Question(**q) for q in questions_raw]
    return papers, questions
