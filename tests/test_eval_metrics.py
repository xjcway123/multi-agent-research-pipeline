"""Tests for the cheap eval metrics (no LLM calls)."""
import pytest

from app.models.schemas import CritiqueReport, CritiqueVerdict, ResearchReport, Source
from eval.metrics import (
    TopicResult,
    TopicSpec,
    aggregate,
    citation_grounding,
    citation_validity,
    evaluate,
    routing_precision,
    source_diversity,
)


def _report(
    *,
    text: str = "",
    sources: list[Source] | None = None,
    critique: CritiqueReport | None = None,
) -> ResearchReport:
    return ResearchReport(
        topic="t",
        summary="s",
        key_findings=["k"],
        full_report=text,
        sources=sources or [],
        word_count=len(text.split()),
        critique=critique,
    )


def test_citation_validity_perfect():
    r = _report(
        text="Foo [1] bar [2] baz.",
        sources=[Source(title="a", url="https://a", snippet=""),
                 Source(title="b", url="https://b", snippet="")],
    )
    assert citation_validity(r) == 1.0


def test_citation_validity_catches_phantom_citation():
    r = _report(
        text="Foo [1] bar [3] baz.",  # [3] is phantom, only 1 source
        sources=[Source(title="a", url="https://a", snippet="")],
    )
    assert citation_validity(r) == 0.5


def test_citation_validity_no_citations_is_one():
    r = _report(text="Plain prose with no citations.", sources=[])
    assert citation_validity(r) == 1.0


def test_citation_validity_zero_sources_with_markers():
    r = _report(text="Bogus [1] [2] markers.", sources=[])
    assert citation_validity(r) == 0.0


def test_citation_grounding_returns_critic_score():
    crit = CritiqueReport(score=0.42, verdicts=[])
    r = _report(text="x", critique=crit)
    assert citation_grounding(r) == 0.42


def test_citation_grounding_none_when_disabled():
    r = _report(text="x", critique=None)
    assert citation_grounding(r) is None


def test_source_diversity_unique_domains():
    sources = [
        Source(title="a", url="https://arxiv.org/abs/1", snippet=""),
        Source(title="b", url="https://arxiv.org/abs/2", snippet=""),
        Source(title="c", url="https://github.com/x/y", snippet=""),
        Source(title="d", url="https://en.wikipedia.org/wiki/Z", snippet=""),
    ]
    r = _report(sources=sources)
    # 3 distinct domains out of 4 sources
    assert source_diversity(r) == 0.75


def test_source_diversity_empty():
    assert source_diversity(_report()) == 0.0


def test_source_diversity_strips_www():
    sources = [
        Source(title="a", url="https://www.example.com/1", snippet=""),
        Source(title="b", url="https://example.com/2", snippet=""),
    ]
    r = _report(sources=sources)
    # Both should collapse to "example.com" → 1 domain / 2 sources
    assert source_diversity(r) == 0.5


def test_routing_precision_full_hit():
    # "what is" + library + best → wikipedia + github
    assert routing_precision(
        "What is the best vector database library",
        ["wikipedia", "github"],
    ) == 1.0


def test_routing_precision_partial_miss():
    # "what is GraphRAG" → wikipedia only; arxiv expected too
    assert routing_precision("What is GraphRAG", ["wikipedia", "arxiv"]) == 0.5


def test_routing_precision_empty_expectation_is_vacuous():
    assert routing_precision("anything", []) == 1.0


def test_evaluate_combines_metrics():
    r = _report(
        text="Foo [1] bar.",
        sources=[Source(title="a", url="https://a.com", snippet="")],
        critique=CritiqueReport(score=0.9, verdicts=[]),
    )
    spec = TopicSpec(topic="What is X", category="concept-define", expected_sources=["wikipedia"])
    result = evaluate(spec, r, elapsed_seconds=12.3)
    assert result.citation_validity == 1.0
    assert result.citation_grounding == 0.9
    assert result.routing_precision == 1.0
    assert result.word_count == 3
    assert result.elapsed_seconds == 12.3


def test_aggregate_averages_and_handles_nones():
    results = [
        TopicResult(
            topic="a", category="x", citation_validity=1.0, citation_grounding=0.8,
            source_diversity=0.5, routing_precision=1.0, word_count=100,
            elapsed_seconds=10.0,
        ),
        TopicResult(
            topic="b", category="x", citation_validity=0.5, citation_grounding=None,
            source_diversity=1.0, routing_precision=0.5, word_count=200,
            elapsed_seconds=20.0,
        ),
    ]
    agg = aggregate(results)
    assert agg["n_topics"] == 2
    assert agg["avg_citation_validity"] == pytest.approx(0.75)
    # Grounding average skips the None
    assert agg["avg_citation_grounding"] == pytest.approx(0.8)
    assert agg["avg_word_count"] == 150
