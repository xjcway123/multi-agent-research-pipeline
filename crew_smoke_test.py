"""Smoke test: can CrewAI run with ChatClaudeCode (Claude CLI subscription)?

CrewAI 1.14's Agent.llm accepts `str | BaseLLM | None`. Strings get routed
through LiteLLM (paid API). To use the Claude CLI subscription, we need a
custom subclass of crewai.llms.base_llm.BaseLLM that delegates to our
existing langchain-claude-code ChatClaudeCode wrapper.

Run:
    python crew_smoke_test.py
"""
from __future__ import annotations

import time
from typing import Any

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import get_claude_llm


class ClaudeCodeCrewLLM(BaseLLM):
    """CrewAI BaseLLM that proxies to ChatClaudeCode (Claude CLI subscription)."""

    llm_type: str = "claude_code"
    model: str = "sonnet"

    # Don't try to validate / serialise the underlying ChatClaudeCode instance.
    _inner: Any = None

    def __init__(self, model: str = "sonnet", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        # Lazily build once per instance.
        self._inner = get_claude_llm(model=model)

    def _to_lc_messages(self, messages: str | list[dict[str, Any]]) -> list:
        """Translate CrewAI messages to LangChain messages."""
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        return out

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        lc_messages = self._to_lc_messages(messages)
        # ChatClaudeCode.invoke shells out to the Claude CLI; sync is fine.
        response = self._inner.invoke(lc_messages)
        return response.content if hasattr(response, "content") else str(response)

    # CrewAI calls these for accounting; safe defaults.
    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 200_000  # Sonnet 4.x context window


def main() -> None:
    print("Building CrewAI agent with ChatClaudeCode wrapper...")
    llm = ClaudeCodeCrewLLM(model="sonnet")

    planner = Agent(
        role="Research Planner",
        goal="Break a topic into 3 specific research questions",
        backstory=(
            "You are a thorough research planner who produces tight, "
            "answerable questions instead of broad themes."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    plan_task = Task(
        description=(
            "Topic: {topic}\n\n"
            "Produce a numbered list of exactly 3 specific research questions "
            "that, if answered, would let someone write a focused brief on the topic. "
            "Output ONLY the numbered list, no preamble."
        ),
        expected_output="A numbered list of 3 specific research questions.",
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[plan_task], verbose=True)

    print("\nRunning crew with topic='What is GraphRAG?'\n")
    start = time.perf_counter()
    result = crew.kickoff(inputs={"topic": "What is GraphRAG?"})
    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 60}")
    print(f"Wall-clock: {elapsed:.1f}s")
    print(f"Result type: {type(result).__name__}")
    print(f"\nResult.raw:\n{result.raw}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
