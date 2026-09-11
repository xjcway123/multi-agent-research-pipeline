import asyncio
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import PipelineState
from app.observability import observe, trace_llm, update_current
from app.tools.router import explain_routing, route_sources
from app.tools.search import (
    SearchResult,
    format_results_for_prompt,
)
from app.tools.sources import search_many

logger = logging.getLogger(__name__)

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a thorough researcher. Synthesize the provided multi-source "
        "search results into a factual, well-cited answer to the question. "
        "When citing facts, reference sources as [1], [2], etc., matching "
        "the numbered results provided. Stay grounded — do not invent facts "
        "that aren't supported by the sources. Mention the source type "
        "(paper, repo, encyclopedia, discussion) when it adds context."
    ),
    (
        "user",
        "Question: {question}\n\n"
        "Sources used: {sources_used}\n\n"
        "Search results:\n{search_context}\n\n"
        "Write a focused 3-5 sentence answer with inline [1], [2] citations."
    ),
])


def _strip_numbering(line: str) -> str:
    return re.sub(r"^\s*[\d]+[\.\)]\s*", "", line).strip()


def _flatten_results(grouped: dict[str, list[SearchResult]]) -> list[SearchResult]:
    """Flatten {source: [results]} into a single ordered list, deduped by URL."""
    flat: list[SearchResult] = []
    seen: set[str] = set()
    for source, items in grouped.items():
        for item in items:
            if item.url and item.url not in seen:
                seen.add(item.url)
                flat.append(item)
    return flat


async def _research_one_question(
    question: str,
    llm,
    topic: str | None = None,
) -> tuple[str, list[SearchResult], list[str]]:
    """Route → multi-source search → LLM synthesis. Returns (note, sources, source_names)."""
    cleaned = _strip_numbering(question)
    chosen_sources = route_sources(cleaned, topic=topic)

    grouped = await search_many(cleaned, chosen_sources, max_per_source=3)
    flat = _flatten_results(grouped)

    if not flat:
        # Fallback: try DDG directly
        from app.tools.search import web_search_async
        flat = await web_search_async(cleaned, max_results=4)

    search_context = format_results_for_prompt(flat[:8])

    chain = RESEARCH_PROMPT | llm
    payload = {
        "question": cleaned,
        "sources_used": ", ".join(chosen_sources),
        "search_context": search_context,
    }
    # ChatClaudeCode's ainvoke spawns a CLI subprocess via anyio.open_process,
    # which fails inside uvicorn's running event loop on Windows ("Failed to
    # start Claude Code"). Pushing the call to a worker thread gives it its
    # own loop, mirroring how LangGraph runs the (sync) planner node.
    try:
        with trace_llm(
            "researcher.synthesize",
            model="claude-sonnet-4-5",
            input={"question": cleaned, "sources_used": chosen_sources},
            metadata={"num_results": len(flat[:8]), "routing": chosen_sources},
        ):
            response = await asyncio.to_thread(chain.invoke, payload)
            update_current(output=response.content)
    except Exception as exc:
        logger.exception("LLM invoke failed for question %r", cleaned)
        raise

    note = f"**{cleaned}**\n_Sources: {explain_routing(cleaned, chosen_sources)}_\n{response.content}"
    return note, flat, chosen_sources


@observe(name="researcher", as_type="agent", capture_input=False, capture_output=False)
async def _run_researcher_async(state: PipelineState) -> PipelineState:
    from app.config import LLM_BACKEND, get_llm

    llm = get_llm(temperature=0.4)
    questions = [q for q in state.get("plan", []) if q.strip()]

    if not questions:
        state["research_notes"] = []
        state["sources"] = []
        state["routing"] = []
        return state

    topic = state.get("topic")

    # Claude CLI can't be spawned concurrently — running parallel subprocesses
    # races on auth/setup and every call fails with "Failed to start Claude Code".
    # Serialize when on the Claude backend; keep parallelism for HTTP-API LLMs.
    concurrency = 1 if LLM_BACKEND == "claude" else len(questions)
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(q):
        async with sem:
            return await _research_one_question(q, llm, topic=topic)

    results = await asyncio.gather(
        *[_guarded(q) for q in questions], return_exceptions=True
    )

    notes: list[str] = []
    all_sources: list[dict] = []
    seen_urls: set[str] = set()
    routing_log: list[dict] = []

    for q, item in zip(questions, results):
        if isinstance(item, Exception):
            logger.warning("research subtask failed: %s", item)
            continue
        note, sources, source_names = item
        notes.append(note)
        routing_log.append({
            "question": _strip_numbering(q),
            "sources": source_names,
            "found": len(sources),
        })
        for s in sources:
            if s.url and s.url not in seen_urls:
                seen_urls.add(s.url)
                all_sources.append(s.to_dict())

    state["research_notes"] = notes
    state["sources"] = all_sources
    state["routing"] = routing_log
    return state


def run_researcher(state: PipelineState) -> PipelineState:
    """Sync wrapper — only used if graph runs synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.ensure_future(_run_researcher_async(state))
        return loop.run_until_complete(future)
    return asyncio.run(_run_researcher_async(state))


async def arun_researcher(state: PipelineState) -> PipelineState:
    return await _run_researcher_async(state)
