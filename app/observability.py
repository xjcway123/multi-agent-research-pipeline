"""Langfuse observability — agent + LLM-call tracing.

Designed to be **safe to import unconditionally**. When the LANGFUSE keys
aren't set in the environment, every helper degrades to a no-op so the
rest of the codebase doesn't need conditional logic.

Usage:
    from app.observability import observe, trace_llm, update_current

    @observe(as_type="agent", name="planner")
    def run_planner(state): ...

    with trace_llm("planner.invoke", model="claude-sonnet-4.5"):
        response = chain.invoke(payload)
        update_current(output=response.content)
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_ENABLED: bool = bool(_PUBLIC and _SECRET)

_client: Any = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse, observe as _lf_observe  # type: ignore

        _client = Langfuse(
            public_key=_PUBLIC,
            secret_key=_SECRET,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "dev"),
            release=os.getenv("LANGFUSE_RELEASE", "multi-agent-pipeline@dev"),
        )
        observe = _lf_observe  # re-export the real decorator
        logger.info("Langfuse tracing enabled (host=%s)", os.getenv("LANGFUSE_HOST", "cloud"))
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Langfuse setup failed, falling back to no-op: %s", e)
        LANGFUSE_ENABLED = False
        _client = None


if not LANGFUSE_ENABLED:

    def observe(*args: Any, **kwargs: Any) -> Any:
        """Pass-through decorator when Langfuse keys aren't set.

        Supports both @observe and @observe(name=..., as_type=...).
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator


@contextmanager
def trace_llm(
    name: str,
    *,
    model: str = "",
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[None]:
    """Open a generation span around a single LLM invocation. No-ops if disabled."""
    if not LANGFUSE_ENABLED or _client is None:
        yield
        return

    cm = _client.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model or None,
        input=input,
        metadata=metadata or {},
    )
    try:
        with cm:
            yield
    except Exception:
        # The span context manager records exceptions automatically; we
        # just need to re-raise.
        raise


def update_current(**kwargs: Any) -> None:
    """Enrich the currently-active observation (output, metadata, level, ...)."""
    if not LANGFUSE_ENABLED or _client is None:
        return
    try:
        _client.update_current_observation(**kwargs)
    except Exception as e:  # pragma: no cover
        logger.debug("Langfuse update_current failed: %s", e)


def update_current_trace(**kwargs: Any) -> None:
    """Enrich the top-level trace (e.g. session_id, user_id, tags)."""
    if not LANGFUSE_ENABLED or _client is None:
        return
    try:
        _client.update_current_trace(**kwargs)
    except Exception as e:  # pragma: no cover
        logger.debug("Langfuse update_current_trace failed: %s", e)


def flush() -> None:
    """Push pending events. Call at app shutdown to avoid losing data."""
    if not LANGFUSE_ENABLED or _client is None:
        return
    try:
        _client.flush()
    except Exception as e:  # pragma: no cover
        logger.debug("Langfuse flush failed: %s", e)


__all__ = [
    "LANGFUSE_ENABLED",
    "observe",
    "trace_llm",
    "update_current",
    "update_current_trace",
    "flush",
]
