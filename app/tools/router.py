"""Source routing — picks which research sources to use per query.

Rule-based for speed/determinism. Each query gets a small set of sources
(2-4) so we don't hammer all 5 APIs for trivial questions.
"""

from __future__ import annotations

import re

ALL_SOURCES = ["arxiv", "wikipedia", "github", "hackernews", "duckduckgo"]

# Keyword → source signal
_PATTERNS = {
    "arxiv": [
        r"\bpaper\b", r"\bresearch\b", r"\bstate of the art\b", r"\bSOTA\b",
        r"\btransformer\b", r"\bneural\b", r"\barchitecture\b",
        r"\battention mechanism\b", r"\bembedding\b", r"\bfine.?tuning\b",
        r"\b(diffusion|GAN|VAE|RLHF|DPO)\b", r"\bbenchmark\b",
        r"\balgorithm\b", r"\bablation\b",
    ],
    "wikipedia": [
        r"^what is\b", r"\bdefine\b", r"\bdefinition of\b",
        r"\bexplain\b", r"\bconcept of\b", r"\bhistory of\b",
        r"\bwho (is|was)\b", r"\boverview of\b",
    ],
    "github": [
        r"\blibrary\b", r"\bframework\b", r"\btool(s|kit)?\b",
        r"\bbest\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
        r"\bimplementation\b", r"\bopen.?source\b", r"\brepo\b",
        r"\bSDK\b", r"\bAPI\b",
    ],
    "hackernews": [
        r"\blatest\b", r"\btrend(s|ing)?\b", r"\bnews\b",
        r"\b202[5-9]\b", r"\bopinion\b", r"\bdiscussion\b",
        r"\bcurrent\b", r"\brecent\b", r"\bemerging\b",
    ],
}

_COMPILED_PATTERNS = {
    source: re.compile(
        "|".join(f"(?:{pattern})" for pattern in patterns),
        flags=re.IGNORECASE,
    )
    for source, patterns in _PATTERNS.items()
}

_FALLBACK_SOURCES = ("wikipedia", "duckduckgo")

_SOURCE_REASONS = {
    "arxiv": "academic papers",
    "wikipedia": "concept definition",
    "github": "code/tools",
    "hackernews": "trends/discussion",
    "duckduckgo": "general web",
}


def route_sources(
    query: str,
    topic: str | None = None,
    always_include_web: bool = True,
) -> list[str]:
    """Return a list of sources to query for this question.

    The original `topic` (if provided) is matched against the patterns alongside
    the sub-question, so topic-level intent (e.g. "best vector database
    libraries") propagates even when the planner's sub-questions don't repeat
    the trigger keywords.

    DuckDuckGo is included by default as a general fallback unless caller
    sets always_include_web=False. arXiv and Wikipedia are skipped if
    nothing else matches them — we only use them when they're a good fit.
    """
    haystack = query + " \n " + (topic or "")
    chosen: list[str] = []
    chosen_set: set[str] = set()

    def _append(source: str) -> None:
        if source not in chosen_set:
            chosen.append(source)
            chosen_set.add(source)

    for source, pattern in _COMPILED_PATTERNS.items():
        if pattern.search(haystack):
            _append(source)

    if always_include_web:
        _append("duckduckgo")

    # Always seed with at least 2 sources for redundancy
    if len(chosen) < 2:
        fallback_sources = (
            _FALLBACK_SOURCES
            if always_include_web
            else tuple(source for source in _FALLBACK_SOURCES if source != "duckduckgo")
        )
        for fallback in fallback_sources:
            _append(fallback)
            if len(chosen) >= 2:
                break
    return chosen


def explain_routing(query: str, sources: list[str]) -> str:
    """Human-readable explanation of why these sources were picked."""
    return ", ".join(f"{s} ({_SOURCE_REASONS.get(s, '?')})" for s in sources)
