import os
import shutil

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

CLAUDE_CLI_AVAILABLE = bool(shutil.which("claude"))

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-5-turbo")
# "glm" (default) uses Z.ai's GLM-5-turbo for researcher+writer.
# "claude" routes researcher+writer through the Claude CLI subscription
# (free for the user but slower because each call shells out).
LLM_BACKEND = os.getenv("LLM_BACKEND", "glm").lower()


def get_llm(temperature: float = 0.3):
    """Worker LLM (researcher + writer). Backend is env-driven."""
    if LLM_BACKEND == "claude":
        return get_claude_llm(model="sonnet")
    return _get_openai_llm(temperature)


def _get_openai_llm(temperature: float = 0.3) -> ChatOpenAI:
    """OpenAI-compatible client (GLM, OpenRouter, or real OpenAI)."""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


def get_claude_llm(model: str = "sonnet"):
    """Claude via CLI/subscription. `model` accepts 'sonnet', 'opus', 'haiku'.

    Only works when a logged-in Claude CLI is on PATH — i.e. local dev.
    Cloud deployments should use the OpenAI-compatible backend instead.
    """
    from langchain_claude_code import ChatClaudeCode
    return ChatClaudeCode(model=model, permission_mode="default")


def get_planner_llm(use_opus: bool = False):
    """Planner LLM. Falls back to the worker backend when Claude CLI isn't
    available (e.g. on a remote host with no logged-in subscription).

    Local dev with Claude CLI → Claude Sonnet/Opus via subscription.
    Cloud (LLM_BACKEND != 'claude' AND no Claude CLI) → same model as workers.
    """
    if LLM_BACKEND == "claude":
        return get_claude_llm(model="opus" if use_opus else "sonnet")
    return _get_openai_llm(temperature=0.3)
