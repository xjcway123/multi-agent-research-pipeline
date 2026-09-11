import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Synchronous DuckDuckGo search. Returns list of SearchResult.

    Forces region=wt-wt (worldwide) so the user's local IP doesn't bias DDG
    toward localized e-commerce junk (e.g. German shopping pages for English
    research queries).
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            # Over-fetch and then filter so we still hit max_results after dropping junk.
            raw = list(
                ddgs.text(
                    query,
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=max_results * 2,
                )
            )
        out: list[SearchResult] = []
        for r in raw:
            url = r.get("href", "") or r.get("url", "")
            if _looks_like_junk(url):
                continue
            out.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("body", ""),
                )
            )
            if len(out) >= max_results:
                break
        return out
    except Exception as e:
        logger.warning("Web search failed for query %r: %s", query, e)
        return []


_JUNK_TLDS = (".de", ".fr", ".it", ".es", ".ru", ".cn", ".jp", ".kr", ".pl", ".tr", ".br")
_JUNK_HOSTS = ("amazon.", "ebay.", "alibaba.", "aliexpress.", "etsy.", "walmart.")


def _looks_like_junk(url: str) -> bool:
    """Heuristically reject geo-localized shopping/e-commerce results."""
    if not url:
        return True
    lower = url.lower()
    if any(host in lower for host in _JUNK_HOSTS):
        return True
    # Only reject foreign-TLD if path is suspiciously short (homepage/category) —
    # academic/government pages on those TLDs usually have deeper paths.
    try:
        from urllib.parse import urlparse

        parsed = urlparse(lower)
        host = parsed.netloc
        if host.endswith(_JUNK_TLDS) and len(parsed.path.strip("/")) < 20:
            return True
    except Exception:
        pass
    return False


async def web_search_async(query: str, max_results: int = 5) -> list[SearchResult]:
    """Async wrapper that runs DuckDuckGo search in a thread."""
    return await asyncio.to_thread(web_search, query, max_results)


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """Format search results as numbered context for an LLM prompt."""
    if not results:
        return "(no search results found)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    {r.snippet}")
    return "\n".join(lines)
