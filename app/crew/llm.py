"""CrewAI BaseLLM that delegates to ChatClaudeCode (Claude CLI subscription).

CrewAI 1.x's `Agent.llm` accepts `str | BaseLLM | None`. Strings flow through
LiteLLM and need a paid Anthropic API key. This subclass lets us reuse the
free Claude CLI subscription (same code path the LangGraph engine uses)
without any LangChain↔CrewAI cross-pollination beyond message formatting.
"""
from __future__ import annotations

from typing import Any

from crewai.llms.base_llm import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class ClaudeCodeCrewLLM(BaseLLM):
    """Bridges CrewAI's BaseLLM contract to ChatClaudeCode."""

    llm_type: str = "claude_code"
    model: str = "sonnet"
    _inner: Any = None

    def __init__(self, model: str = "sonnet", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        # Lazy import so importing this module doesn't trigger Claude CLI checks.
        from app.config import get_claude_llm

        self._inner = get_claude_llm(model=model)

    @staticmethod
    def _to_lc_messages(messages: str | list[dict[str, Any]]) -> list:
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]
        out: list = []
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
        response = self._inner.invoke(lc_messages)
        return response.content if hasattr(response, "content") else str(response)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 200_000
