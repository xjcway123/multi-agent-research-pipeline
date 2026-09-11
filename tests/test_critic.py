"""Tests for the Critic agent — claim extraction, verdict parsing, scoring."""
import json

import pytest

from app.agents.critic import (
    REPAIR_THRESHOLD,
    _parse_verdicts,
    _score,
    extract_claims,
    needs_repair,
)
from app.models.schemas import CritiqueReport, CritiqueVerdict


def test_extract_claims_pulls_sentences_with_citations():
    report = (
        "## Key Findings\n"
        "- LangGraph models the pipeline as a typed state machine [1].\n"
        "- CrewAI uses Agents and Tasks for sequential orchestration [2][3].\n"
        "\n"
        "## Detailed Analysis\n"
        "Both engines produce the same SSE shape. The Researcher node is shared [1]."
    )
    claims = extract_claims(report)
    assert len(claims) == 3
    assert claims[0][1] == [1]
    assert claims[1][1] == [2, 3]
    assert claims[2][1] == [1]


def test_extract_claims_skips_headers_and_sentences_without_citations():
    report = (
        "# Title without citation\n"
        "## Subheading [1] inside a header should still be skipped\n"
        "This sentence has no citation.\n"
        "This sentence does [4]."
    )
    claims = extract_claims(report)
    assert len(claims) == 1
    assert claims[0][1] == [4]


def test_parse_verdicts_handles_clean_json():
    claims = [("claim a", [1]), ("claim b", [2])]
    raw = json.dumps(
        [
            {"index": 0, "verdict": "supported", "reason": "snippet backs claim"},
            {"index": 1, "verdict": "unsupported", "reason": "snippet unrelated"},
        ]
    )
    verdicts = _parse_verdicts(raw, claims)
    assert [v.verdict for v in verdicts] == ["supported", "unsupported"]
    assert verdicts[0].claim == "claim a"
    assert verdicts[1].citation_indices == [2]


def test_parse_verdicts_strips_code_fence():
    claims = [("only claim", [1])]
    raw = "```json\n[{\"index\": 0, \"verdict\": \"partial\", \"reason\": \"meh\"}]\n```"
    verdicts = _parse_verdicts(raw, claims)
    assert verdicts[0].verdict == "partial"


def test_parse_verdicts_falls_back_to_partial_on_garbage():
    claims = [("a", [1]), ("b", [2])]
    verdicts = _parse_verdicts("not json at all {{}}", claims)
    assert len(verdicts) == 2
    # All default to partial when parse fails — never crashes the pipeline
    assert all(v.verdict == "partial" for v in verdicts)


def test_parse_verdicts_rejects_invalid_verdict_strings():
    claims = [("c", [1])]
    raw = json.dumps([{"index": 0, "verdict": "totally-made-up", "reason": ""}])
    verdicts = _parse_verdicts(raw, claims)
    assert verdicts[0].verdict == "partial"


def test_score_arithmetic():
    verdicts = [
        CritiqueVerdict(claim="a", citation_indices=[1], verdict="supported"),
        CritiqueVerdict(claim="b", citation_indices=[2], verdict="partial"),
        CritiqueVerdict(claim="c", citation_indices=[3], verdict="unsupported"),
        CritiqueVerdict(claim="d", citation_indices=[4], verdict="no_citation"),
    ]
    # (1.0 + 0.5 + 0 + 0) / 4 = 0.375
    assert _score(verdicts) == pytest.approx(0.375)


def test_score_empty_is_vacuously_one():
    assert _score([]) == 1.0


def test_needs_repair_triggers_when_score_low_and_unsupported_present():
    state = {
        "enable_critic": True,
        "critique_iterations": 0,
        "critique": {
            "score": 0.5,
            "verdicts": [{"verdict": "unsupported"}, {"verdict": "supported"}],
        },
    }
    assert needs_repair(state) == "repair"


def test_needs_repair_skips_when_already_repaired():
    state = {
        "enable_critic": True,
        "critique_iterations": 1,  # already repaired once
        "critique": {
            "score": 0.4,
            "verdicts": [{"verdict": "unsupported"}],
        },
    }
    assert needs_repair(state) == "end"


def test_needs_repair_skips_when_critic_disabled():
    state = {
        "enable_critic": False,
        "critique_iterations": 0,
        "critique": {"score": 0.1, "verdicts": [{"verdict": "unsupported"}]},
    }
    assert needs_repair(state) == "end"


def test_needs_repair_skips_when_above_threshold():
    state = {
        "enable_critic": True,
        "critique_iterations": 0,
        "critique": {
            "score": REPAIR_THRESHOLD + 0.01,
            "verdicts": [{"verdict": "unsupported"}],
        },
    }
    assert needs_repair(state) == "end"


def test_critique_report_serialization_roundtrip():
    report = CritiqueReport(
        score=0.75,
        verdicts=[
            CritiqueVerdict(claim="x", citation_indices=[1], verdict="supported"),
        ],
        repaired=True,
        iterations=2,
    )
    raw = report.model_dump()
    revived = CritiqueReport.model_validate(raw)
    assert revived.score == 0.75
    assert revived.repaired is True
    assert revived.supported_count == 1
    assert revived.unsupported_count == 0
