"""Metrics for evaluating a single pipeline run against a topic spec.

The cheap metrics (citation_validity, source_diversity, routing_precision,
word_count) are pure functions over the ResearchReport + the topic's
expected categories. citation_grounding reuses the Critic's score, which
the pipeline already computes when enable_critic=True.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.models.schemas import ResearchReport
from app.tools.router import route_sources

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class TopicSpec:
    topic: str
    category: str
    expected_sources: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TopicResult:
    topic: str
    category: str
    citation_validity: float
    citation_grounding: float | None  # None if critic disabled
    source_diversity: float
    routing_precision: float
    word_count: int
    elapsed_seconds: float
    repaired: bool = False
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "topic": self.topic,
            "category": self.category,
            "citation_validity": round(self.citation_validity, 3),
            "citation_grounding": (
                round(self.citation_grounding, 3)
                if self.citation_grounding is not None
                else None
            ),
            "source_diversity": round(self.source_diversity, 3),
            "routing_precision": round(self.routing_precision, 3),
            "word_count": self.word_count,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "repaired": self.repaired,
        }


def citation_validity(report: ResearchReport) -> float:
    """% of [N] markers in the report that point to an actual source index.

    Free, no LLM. A score of 1.0 means every numbered citation in the text
    has a matching entry in report.sources.
    """
    text = report.full_report or ""
    markers = [int(m) for m in _CITATION_RE.findall(text)]
    if not markers:
        return 1.0  # nothing to verify
    n_sources = len(report.sources)
    if n_sources == 0:
        return 0.0
    valid = sum(1 for m in markers if 1 <= m <= n_sources)
    return valid / len(markers)


def citation_grounding(report: ResearchReport) -> float | None:
    """The Critic's score, if available. None when critic was disabled."""
    if report.critique is None:
        return None
    return report.critique.score


def source_diversity(report: ResearchReport) -> float:
    """Distinct domains / total sources. Higher = drawing from more places."""
    if not report.sources:
        return 0.0
    domains = set()
    for s in report.sources:
        url = s.url or ""
        # crude domain extraction — host between // and the first /
        if "://" in url:
            host = url.split("://", 1)[1].split("/", 1)[0].lower()
            if host.startswith("www."):
                host = host[4:]
            domains.add(host)
    return len(domains) / len(report.sources)


def routing_precision(topic: str, expected: Iterable[str]) -> float:
    """Fraction of expected sources actually chosen by the router for this topic.

    The router runs on the original topic string here as a proxy for the
    planner's sub-questions (the planner output isn't returned by the API,
    only the final report). expected is a subset of:
    arxiv / wikipedia / github / hackernews / duckduckgo.
    """
    expected_set = {s.lower() for s in expected}
    if not expected_set:
        return 1.0  # nothing expected → vacuously satisfied
    chosen = set(route_sources(topic))
    hit = expected_set & chosen
    return len(hit) / len(expected_set)


def evaluate(
    spec: TopicSpec,
    report: ResearchReport,
    elapsed_seconds: float,
) -> TopicResult:
    return TopicResult(
        topic=spec.topic,
        category=spec.category,
        citation_validity=citation_validity(report),
        citation_grounding=citation_grounding(report),
        source_diversity=source_diversity(report),
        routing_precision=routing_precision(spec.topic, spec.expected_sources),
        word_count=report.word_count,
        elapsed_seconds=elapsed_seconds,
        repaired=bool(report.critique and report.critique.repaired),
        notes=spec.notes,
    )


def aggregate(results: list[TopicResult]) -> dict:
    """Aggregate metrics across all topics. Skips Nones."""
    if not results:
        return {}

    def _avg(key: str) -> float | None:
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_topics": len(results),
        "avg_citation_validity": _avg("citation_validity"),
        "avg_citation_grounding": _avg("citation_grounding"),
        "avg_source_diversity": _avg("source_diversity"),
        "avg_routing_precision": _avg("routing_precision"),
        "avg_word_count": _avg("word_count"),
        "avg_elapsed_seconds": _avg("elapsed_seconds"),
        "repaired_count": sum(1 for r in results if r.repaired),
    }
