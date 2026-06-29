"""End-to-end LLM-as-judge answer quality eval.

For each (retriever, question):

    1. Build context from retriever's returned paragraphs.
    2. Generate an answer via the LLM grounded in that context.
    3. Judge the answer against the gold evidence paragraphs.

The judge sees the gold evidence (already linked via gold_paragraph_ids)
and the predicted answer, and produces a 0..1 correctness score plus a
short rationale. This avoids needing to extract the textual gold answer
from QASPER's mixed-type answer fields.

Why this is the right next experiment:
DRAG-Subtree-tight wins relevance_density per token (0.133 vs
hybrid_rrf_flat 0.103) but loses Recall@5 (0.124 vs 0.387). If "coherent
context is more useful than disjoint top-K" — the central DRAG-Subtree
claim — that has to translate into better LLM answers at matched context
budget. This eval is the verdict.

Reads `benchmarks/results/raw.jsonl` + a QASPER slice directory.
Writes `benchmarks/results/qa_eval.jsonl` with one row per (retriever,
question) carrying answer + score + rationale.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from rag_lib.utils import llm_call

from .datasets import load_slice


RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / "cache"
SLICE_DIR_DEFAULT = CACHE_DIR / "qasper_p10_q60_s0"


# Retrievers to evaluate. Three points on the (recall, density, coherence)
# Pareto: flat top-5, hierarchical cluster summaries, coherent subtree.
DEFAULT_RETRIEVERS = ("hybrid_rrf_flat", "raptor_collapsed", "drag_subtree_tight")

# Hard cap on context to keep judging fair when retrievers return very
# different ctx_sizes. Roughly 10 typical paragraphs.
CONTEXT_PARA_CAP = 12


ANSWER_PROMPT = """# Task
You are a research assistant. Given a question and a small set of excerpts from one or more scientific papers, write a concise answer grounded ONLY in the excerpts.

# Rules
- Do not invent facts that aren't in the excerpts.
- If the excerpts don't answer the question, say "The provided excerpts do not answer this question."
- Keep the answer under 80 words. Be specific.
- Do NOT preface with "Based on the excerpts" / "According to the paper".

# Output
Plain text answer. No JSON, no markdown.
"""


JUDGE_PROMPT = """# Task
You grade a research assistant's answer against the ground-truth evidence.

# Inputs
- question: the question that was asked.
- gold_evidence: paragraphs from the paper that contain the ground-truth answer.
- predicted_answer: what the assistant produced from a different (potentially noisy or partial) set of excerpts.

# Scoring
Return a single JSON object with:
- "score": float in [0.0, 1.0]
    1.0   answer is fully correct and well-supported by gold_evidence
    0.75  answer is correct on the main fact but missing supporting detail OR has minor extra noise
    0.5   answer is partially correct — gets the right entity/value but misses key qualification, or vice versa
    0.25  answer is mostly off but touches a relevant aspect
    0.0   answer is wrong, unrelated, or a refusal when the evidence does contain the answer
- "rationale": 1 short sentence explaining the score.

Calibrate strictly. Reward only answers actually grounded in gold_evidence content.
"""


class AnswerOutput(BaseModel):
    """Free-form text wrapped so we can route through the same llm_call schema."""

    answer: str = Field(description="The plain-text answer, under 80 words.")


class JudgeOutput(BaseModel):
    score: float = Field(description="0.0 to 1.0, see calibration rubric in the system prompt.")
    rationale: str = Field(description="One short sentence justification.")


def _paragraph_lookup(slice_dir: Path) -> dict[str, str]:
    """Build paragraph_id -> text dict from the cached QASPER slice."""
    papers, _ = load_slice(slice_dir)
    return {
        para.paragraph_id: para.text
        for paper in papers
        for para in paper.paragraphs
    }


def _build_context(
    retrieved: list[str], paragraph_text: dict[str, str], cap: int = CONTEXT_PARA_CAP
) -> tuple[str, int]:
    """Concatenate retrieved paragraphs into one context block (capped).
    Returns (context_string, num_paragraphs_used)."""
    used = retrieved[:cap]
    blocks: list[str] = []
    for pid in used:
        text = paragraph_text.get(pid)
        if text:
            blocks.append(f"[{pid}]\n{text}")
    return "\n\n---\n\n".join(blocks), len(blocks)


def _build_gold_context(
    gold_ids: list[str], paragraph_text: dict[str, str]
) -> str:
    blocks: list[str] = []
    for pid in gold_ids:
        text = paragraph_text.get(pid)
        if text:
            blocks.append(f"[{pid}]\n{text}")
    return "\n\n---\n\n".join(blocks) if blocks else "(no gold evidence available)"


def _call_with_retries(messages: list[dict], schema_cls):
    """Three-attempt wrapper around llm_call returning a validated pydantic object,
    or None if all attempts fail."""
    for _ in range(3):
        try:
            out_str, _ = llm_call(messages, schema_cls)
            return schema_cls.model_validate_json(out_str)
        except Exception as exc:
            print(f"  [llm_call retry] {exc!r}", flush=True)
            continue
    return None


def generate_answer(question: str, context: str) -> Optional[str]:
    messages = [
        {"role": "system", "content": ANSWER_PROMPT},
        {
            "role": "user",
            "content": f"# Question\n{question}\n\n# Excerpts\n{context}",
        },
    ]
    out = _call_with_retries(messages, AnswerOutput)
    return out.answer if out is not None else None


def judge_answer(
    question: str, gold_evidence: str, predicted_answer: str
) -> Optional[JudgeOutput]:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"# question\n{question}\n\n"
                f"# gold_evidence\n{gold_evidence}\n\n"
                f"# predicted_answer\n{predicted_answer}"
            ),
        },
    ]
    return _call_with_retries(messages, JudgeOutput)


def main(retrievers: tuple[str, ...] = DEFAULT_RETRIEVERS, slice_dir: Path = SLICE_DIR_DEFAULT) -> int:
    raw_rows = [
        json.loads(l)
        for l in (RESULTS_DIR / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]
    paragraph_text = _paragraph_lookup(slice_dir)
    out_path = RESULTS_DIR / "qa_eval.jsonl"
    out_fh = open(out_path, "w", encoding="utf-8")

    selected = [r for r in raw_rows if r["retriever"] in retrievers]
    total = len(selected)
    print(f"[qa_eval] {total} rows to evaluate "
          f"({len(retrievers)} retrievers x ~35 questions)", flush=True)

    done = 0
    for row in selected:
        retriever = row["retriever"]
        qid = row["question_id"]
        question = row["question"]
        gold_ids = row.get("gold_paragraph_ids") or []
        retrieved = row.get("retrieved") or []

        context, ctx_para_count = _build_context(retrieved, paragraph_text)
        gold_context = _build_gold_context(gold_ids, paragraph_text)

        t0 = time.perf_counter()
        answer = generate_answer(question, context) if context else None
        gen_elapsed = time.perf_counter() - t0
        if answer is None:
            answer = "(generation failed)"

        t1 = time.perf_counter()
        judgement = judge_answer(question, gold_context, answer)
        judge_elapsed = time.perf_counter() - t1

        out_row = {
            "retriever": retriever,
            "question_id": qid,
            "question": question,
            "paper_id": row.get("paper_id"),
            "gold_paragraph_ids": gold_ids,
            "retrieved_count": len(retrieved),
            "ctx_paragraphs": ctx_para_count,
            "answer": answer,
            "judge_score": judgement.score if judgement else None,
            "judge_rationale": judgement.rationale if judgement else None,
            "gen_latency_s": gen_elapsed,
            "judge_latency_s": judge_elapsed,
        }
        out_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_fh.flush()
        done += 1
        if done % 5 == 0 or done == total:
            print(
                f"[qa_eval] {done}/{total} "
                f"(last: {retriever} q={qid[:8]} score={out_row['judge_score']})",
                flush=True,
            )

    out_fh.close()
    print(f"[qa_eval] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
