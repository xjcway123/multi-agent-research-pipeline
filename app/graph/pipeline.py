import re

from langgraph.graph import StateGraph, END

from app.agents.critic import needs_repair, run_critic, run_writer_repair
from app.agents.planner import run_planner
from app.agents.researcher import arun_researcher
from app.agents.writer import run_writer
from app.models.schemas import CritiqueReport, PipelineState, ResearchReport, Source


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("planner", run_planner)
    graph.add_node("researcher", arun_researcher)  # async node
    graph.add_node("writer", run_writer)
    graph.add_node("critic", run_critic)  # async node
    graph.add_node("writer_repair", run_writer_repair)  # async node

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    # Conditional edge — bounce to repair once if critique score is low,
    # otherwise end. The needs_repair function enforces the cap so we
    # can't loop forever.
    graph.add_conditional_edges(
        "critic",
        needs_repair,
        {"repair": "writer_repair", "end": END},
    )
    graph.add_edge("writer_repair", END)

    return graph.compile()


COMPILED_GRAPH = build_graph()


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+(.+)$")


def _extract_key_findings(report: str, max_findings: int = 5) -> list[str]:
    """Pull bullet points out of a markdown report."""
    findings: list[str] = []
    in_findings_section = False

    for line in report.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Detect a "Key Findings" header so we prefer those bullets
        if re.match(r"^#{1,4}\s*key findings", stripped, re.IGNORECASE):
            in_findings_section = True
            continue
        if stripped.startswith("#"):
            in_findings_section = False

        match = _BULLET_RE.match(line)
        if match:
            text = match.group(1).strip().strip("*_`")
            if len(text) > 20:  # skip tiny bullets like single words
                findings.append(text)
                if in_findings_section and len(findings) >= max_findings:
                    break

    return findings[:max_findings]


def _build_summary(report: str, max_chars: int = 400) -> str:
    """Take the first meaningful paragraph as the summary."""
    paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
    for p in paragraphs:
        clean = p.lstrip("#").strip()
        if len(clean) >= 60 and not clean.startswith(("**Executive", "Author")):
            return clean[:max_chars] + ("…" if len(clean) > max_chars else "")
    return report[:max_chars]


def state_to_report(topic: str, final_state: PipelineState) -> ResearchReport:
    report_text = final_state.get("report", "")
    findings = _extract_key_findings(report_text)
    sources_raw = final_state.get("sources", []) or []

    sources = [
        Source(
            title=s.get("title", ""),
            url=s.get("url", ""),
            snippet=s.get("snippet", ""),
        )
        for s in sources_raw
    ]

    critique_dict = final_state.get("critique")
    critique = CritiqueReport.model_validate(critique_dict) if critique_dict else None

    return ResearchReport(
        topic=topic,
        summary=_build_summary(report_text),
        key_findings=findings or ["See full report"],
        full_report=report_text,
        sources=sources,
        word_count=len(report_text.split()),
        critique=critique,
    )


async def run_pipeline(
    topic: str,
    depth: str,
    use_opus_planner: bool = False,
    enable_critic: bool = True,
) -> ResearchReport:
    from app.observability import observe, update_current_trace

    @observe(name="run_pipeline.langgraph", as_type="chain", capture_input=False, capture_output=False)
    async def _traced() -> ResearchReport:
        update_current_trace(
            name="research-pipeline",
            input={
                "topic": topic,
                "depth": depth,
                "use_opus_planner": use_opus_planner,
                "enable_critic": enable_critic,
            },
            metadata={"engine": "langgraph"},
            tags=["langgraph", "research"],
        )
        initial_state: PipelineState = {
            "topic": topic,
            "depth": depth,
            "plan": [],
            "research_notes": [],
            "sources": [],
            "routing": [],
            "report": "",
            "use_opus_planner": use_opus_planner,
            "enable_critic": enable_critic,
            "critique_iterations": 0,
        }
        final_state = await COMPILED_GRAPH.ainvoke(initial_state)
        report = state_to_report(topic, final_state)
        update_current_trace(
            output={
                "word_count": report.word_count,
                "source_count": len(report.sources),
                "critique_score": report.critique.score if report.critique else None,
            },
        )
        return report

    return await _traced()
