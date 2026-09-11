"""Critic agent — verifies that the Writer's citations actually support their claims.

Extracts every sentence containing `[N]` markers, looks up the cited source
snippets, and asks an LLM to judge each claim as supported / partial /
unsupported / no_citation. Aggregates to a single score; the LangGraph
pipeline uses that score to decide whether to bounce to a one-shot repair pass.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import CritiqueReport, CritiqueVerdict, PipelineState
from app.observability import observe, trace_llm, update_current

logger = logging.getLogger(__name__)

# Threshold below which we trigger a repair pass. Set deliberately on the
# stricter side — a report with 25% of citations unsupported is not shippable.
REPAIR_THRESHOLD = 0.75
MAX_REPAIR_ITERATIONS = 1

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Sentence splitter — naive but adequate for markdown reports. We split on
# sentence-ending punctuation followed by whitespace, keeping the punctuation.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a citation auditor. For each claim, the user provides the "
        "claim text and the snippets from its cited sources. Decide whether "
        "the snippets actually support the claim.\n\n"
        "Verdicts:\n"
        "- supported: the snippet(s) clearly back the claim\n"
        "- partial: snippet is related but doesn't fully back the claim\n"
        "- unsupported: snippet does not back the claim or contradicts it\n\n"
        "Respond with ONLY a JSON array of objects, one per claim, each with "
        "keys 'index' (int, the claim index from input), 'verdict' (string), "
        "'reason' (one short sentence). No prose outside the JSON."
    ),
    (
        "user",
        "Claims to audit:\n{claims_block}\n\n"
        "Return the JSON array now."
    ),
])


def extract_claims(report: str) -> list[tuple[str, list[int]]]:
    """Pull (sentence, [cited indices]) tuples from a markdown report.

    Only returns sentences that carry at least one [N] marker. Bullets are
    treated as sentences. Headers are skipped.
    """
    claims: list[tuple[str, list[int]]] = []
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip bullet/list prefixes so the sentence reads clean
        line = re.sub(r"^[-*•]\s+|^\d+\.\s+", "", line)
        # Split into sentences within the line
        for sent in _SENTENCE_RE.split(line):
            sent = sent.strip()
            if not sent:
                continue
            indices = [int(m) for m in _CITATION_RE.findall(sent)]
            if indices:
                claims.append((sent, indices))
    return claims


def _build_claims_block(
    claims: list[tuple[str, list[int]]],
    sources: list[dict],
) -> str:
    """Format claims + their cited snippets into the LLM-facing prompt block."""
    lines: list[str] = []
    for i, (claim, indices) in enumerate(claims):
        lines.append(f"[Claim {i}] {claim}")
        for idx in indices:
            # Citation indices are 1-based in the report
            src_idx = idx - 1
            if 0 <= src_idx < len(sources):
                snippet = (sources[src_idx].get("snippet") or "").strip()
                title = sources[src_idx].get("title", "")
                lines.append(
                    f"  cite[{idx}] {title[:80]}: {snippet[:300]}"
                )
            else:
                lines.append(f"  cite[{idx}] (missing — no source at index {idx})")
        lines.append("")
    return "\n".join(lines)


def _parse_verdicts(
    raw: str,
    claims: list[tuple[str, list[int]]],
) -> list[CritiqueVerdict]:
    """Parse the LLM's JSON array into CritiqueVerdict objects.

    Falls back to 'partial' for any claim the LLM forgot or mis-indexed —
    we never want a parse failure to kill the pipeline.
    """
    # Strip code fences if the model wrapped the JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)

    parsed: list[dict] = []
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            parsed = []
    except json.JSONDecodeError:
        logger.warning("Critic JSON parse failed; falling back to partial verdicts")

    by_index = {p.get("index"): p for p in parsed if isinstance(p, dict)}

    verdicts: list[CritiqueVerdict] = []
    valid = {"supported", "partial", "unsupported", "no_citation"}
    for i, (claim, indices) in enumerate(claims):
        entry = by_index.get(i, {})
        v = entry.get("verdict", "partial")
        if v not in valid:
            v = "partial"
        verdicts.append(
            CritiqueVerdict(
                claim=claim,
                citation_indices=indices,
                verdict=v,
                reason=str(entry.get("reason", ""))[:240],
            )
        )
    return verdicts


def _score(verdicts: list[CritiqueVerdict]) -> float:
    if not verdicts:
        # No claims to verify → vacuously "fine". Don't trigger repair.
        return 1.0
    weights = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "no_citation": 0.0}
    total = sum(weights[v.verdict] for v in verdicts)
    return total / len(verdicts)


async def _critique_async(
    report: str,
    sources: list[dict],
    llm,
) -> CritiqueReport:
    claims = extract_claims(report)
    if not claims:
        return CritiqueReport(score=1.0, verdicts=[], repaired=False, iterations=1)

    claims_block = _build_claims_block(claims, sources)
    chain = CRITIC_PROMPT | llm
    payload = {"claims_block": claims_block}

    with trace_llm(
        "critic.audit",
        model="claude-sonnet-4-5",
        input={"claim_count": len(claims), "source_count": len(sources)},
    ):
        # Mirror researcher's pattern: push to a worker thread so the Claude
        # CLI backend doesn't fight uvicorn's running event loop on Windows.
        response = await asyncio.to_thread(chain.invoke, payload)
        update_current(output=response.content[:2000])

    verdicts = _parse_verdicts(response.content, claims)
    return CritiqueReport(
        score=_score(verdicts),
        verdicts=verdicts,
        repaired=False,
        iterations=1,
    )


@observe(name="critic", as_type="agent", capture_input=False, capture_output=False)
async def run_critic(state: PipelineState) -> PipelineState:
    """LangGraph node — audit the Writer's report, write critique into state."""
    if not state.get("enable_critic", True):
        return state

    from app.config import get_llm

    llm = get_llm(temperature=0.1)
    report = state.get("report", "")
    sources = state.get("sources", []) or []

    critique = await _critique_async(report, sources, llm)
    state["critique"] = critique.model_dump()
    state.setdefault("critique_iterations", 0)
    return state


REPAIR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You previously wrote the report below. The Critic flagged citations "
        "that don't support their claims. For each flagged claim, either "
        "(a) rewrite the claim to be backed by the snippet, or (b) remove the "
        "claim entirely. Do NOT invent new sources. Preserve the report's "
        "structure and untouched paragraphs. Return the full revised report."
    ),
    (
        "user",
        "Original report:\n{report}\n\n"
        "Flagged claims with their cited snippets:\n{flagged_block}\n\n"
        "Return the full revised markdown report now."
    ),
])


async def repair_report(
    report: str,
    critique: CritiqueReport,
    sources: list[dict],
    llm,
) -> str:
    """One-shot repair pass — rewrite only the unsupported claims."""
    flagged = [v for v in critique.verdicts if v.verdict == "unsupported"]
    if not flagged:
        return report

    lines: list[str] = []
    for i, v in enumerate(flagged):
        lines.append(f"[Flagged {i}] {v.claim}")
        lines.append(f"  reason: {v.reason}")
        for idx in v.citation_indices:
            src_idx = idx - 1
            if 0 <= src_idx < len(sources):
                snippet = (sources[src_idx].get("snippet") or "").strip()
                lines.append(f"  cite[{idx}]: {snippet[:300]}")
        lines.append("")
    flagged_block = "\n".join(lines)

    chain = REPAIR_PROMPT | llm
    payload = {"report": report, "flagged_block": flagged_block}

    with trace_llm(
        "writer.repair",
        model="claude-sonnet-4-5",
        input={"flagged_count": len(flagged)},
    ):
        response = await asyncio.to_thread(chain.invoke, payload)
        update_current(output=response.content[:2000])

    return response.content


@observe(name="writer_repair", as_type="agent", capture_input=False, capture_output=False)
async def run_writer_repair(state: PipelineState) -> PipelineState:
    """LangGraph repair node — runs only when critic flagged the report."""
    from app.config import get_llm

    critique_dict = state.get("critique") or {}
    if not critique_dict:
        return state
    critique = CritiqueReport.model_validate(critique_dict)

    llm = get_llm(temperature=0.4)
    revised = await repair_report(
        state.get("report", ""),
        critique,
        state.get("sources", []) or [],
        llm,
    )
    state["report"] = revised

    # Re-critique the revised report so the final score reflects the repair
    rechecked = await _critique_async(revised, state.get("sources", []) or [], llm)
    rechecked.repaired = True
    rechecked.iterations = state.get("critique_iterations", 0) + 2  # original + repair recheck
    state["critique"] = rechecked.model_dump()
    state["critique_iterations"] = state.get("critique_iterations", 0) + 1
    return state


def needs_repair(state: PipelineState) -> str:
    """Conditional edge: 'repair' or 'end'."""
    if not state.get("enable_critic", True):
        return "end"
    critique_dict = state.get("critique") or {}
    score = critique_dict.get("score", 1.0)
    iterations = state.get("critique_iterations", 0)
    has_unsupported = any(
        v.get("verdict") == "unsupported" for v in critique_dict.get("verdicts", [])
    )
    if score < REPAIR_THRESHOLD and has_unsupported and iterations < MAX_REPAIR_ITERATIONS:
        return "repair"
    return "end"
