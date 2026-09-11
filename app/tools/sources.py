"""Free source clients — no API keys required.

Each search function returns list[SearchResult] with consistent shape so the
router can fan out queries in parallel and merge results uniformly.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx

from app.tools.search import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(10.0)
USER_AGENT = "multi-agent-pipeline/1.1 (https://github.com/axon011/multi-agent-pipeline)"


def _truncate(text: str, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def _clean_query(query: str, max_chars: int = 200) -> str:
    """Strip markdown/punctuation/dashes that break upstream search APIs.

    Handles:
      - markdown formatting (** _ ` # > \\)
      - em/en-dashes (— –) that some APIs reject or treat as phrase delimiters
      - trailing question marks (arXiv doesn't need them in the query)
      - collapsed whitespace
      - hard length cap
    """
    cleaned = re.sub(r"[*_`#>\\]", " ", query or "")
    # Replace em/en-dashes with simple spaces; arXiv has been observed to 500 on these.
    cleaned = cleaned.replace("—", " ").replace("–", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip trailing question marks — search APIs don't need them and they
    # occasionally produce stray empty tokens after URL encoding.
    cleaned = cleaned.rstrip("?")
    return cleaned[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# arXiv  —  https://export.arxiv.org/api
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


async def search_arxiv(query: str, max_results: int = 4) -> list[SearchResult]:
    cleaned = _clean_query(query)
    if not cleaned:
        return []
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{quote_plus(cleaned)}"
        f"&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
        results: list[SearchResult] = []
        for entry in root.findall("a:entry", ARXIV_NS):
            title_el = entry.find("a:title", ARXIV_NS)
            summary_el = entry.find("a:summary", ARXIV_NS)
            id_el = entry.find("a:id", ARXIV_NS)
            if title_el is None or id_el is None:
                continue
            authors = [
                (a.find("a:name", ARXIV_NS).text or "")
                for a in entry.findall("a:author", ARXIV_NS)
                if a.find("a:name", ARXIV_NS) is not None
            ][:3]
            author_str = ", ".join(authors)
            title = _truncate(title_el.text or "", 200)
            summary = _truncate(summary_el.text if summary_el is not None else "", 400)
            snippet = f"({author_str}) {summary}" if author_str else summary
            results.append(SearchResult(title=title, url=id_el.text or "", snippet=snippet))
        return results
    except Exception as e:
        logger.warning("arXiv search failed for %r: %s", query, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia  —  https://en.wikipedia.org/w/api.php
# ─────────────────────────────────────────────────────────────────────────────


async def search_wikipedia(query: str, max_results: int = 3) -> list[SearchResult]:
    cleaned = _clean_query(query)
    if not cleaned:
        return []
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": cleaned,
        "srlimit": max_results,
        "format": "json",
        "srprop": "snippet",
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        data = resp.json()
        results: list[SearchResult] = []
        for hit in data.get("query", {}).get("search", []):
            title = hit.get("title", "")
            page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            results.append(
                SearchResult(title=title, url=page_url, snippet=_truncate(snippet, 350))
            )
        return results
    except Exception as e:
        logger.warning("Wikipedia search failed for %r: %s", query, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GitHub  —  https://api.github.com/search/repositories
# Unauthenticated: 60 req/hour. Plenty for a portfolio demo.
# ─────────────────────────────────────────────────────────────────────────────


async def search_github(query: str, max_results: int = 4) -> list[SearchResult]:
    cleaned = _clean_query(query, max_chars=120)
    if not cleaned:
        return []
    url = "https://api.github.com/search/repositories"
    params = {
        "q": cleaned,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    }
    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
        ) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        data = resp.json()
        results: list[SearchResult] = []
        for item in data.get("items", [])[:max_results]:
            title = f"{item.get('full_name', '')} ⭐ {item.get('stargazers_count', 0)}"
            description = item.get("description") or ""
            language = item.get("language") or ""
            snippet = f"[{language}] {description}" if language else description
            results.append(
                SearchResult(
                    title=title,
                    url=item.get("html_url", ""),
                    snippet=_truncate(snippet, 350),
                )
            )
        return results
    except Exception as e:
        logger.warning("GitHub search failed for %r: %s", query, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Hacker News  —  https://hn.algolia.com/api/v1/search
# ─────────────────────────────────────────────────────────────────────────────


async def search_hackernews(query: str, max_results: int = 4) -> list[SearchResult]:
    cleaned = _clean_query(query)
    if not cleaned:
        return []
    url = "https://hn.algolia.com/api/v1/search"
    params = {
        "query": cleaned,
        "hitsPerPage": max_results,
        "tags": "story",
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        data = resp.json()
        results: list[SearchResult] = []
        for hit in data.get("hits", [])[:max_results]:
            title = hit.get("title") or hit.get("story_title") or ""
            link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            points = hit.get("points", 0)
            comments = hit.get("num_comments", 0)
            snippet = f"{points} points · {comments} comments"
            if hit.get("story_text"):
                snippet += f" — {_truncate(hit['story_text'], 200)}"
            results.append(SearchResult(title=title, url=link, snippet=snippet))
        return results
    except Exception as e:
        logger.warning("HN search failed for %r: %s", query, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Registry — name → callable
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_REGISTRY = {
    "arxiv": search_arxiv,
    "wikipedia": search_wikipedia,
    "github": search_github,
    "hackernews": search_hackernews,
}


async def search_one(source: str, query: str, max_results: int = 4) -> list[SearchResult]:
    fn = SOURCE_REGISTRY.get(source)
    if fn is None:
        # Lazy import to avoid circular dep
        if source == "duckduckgo":
            from app.tools.search import web_search_async
            return await web_search_async(query, max_results=max_results)
        logger.warning("Unknown source: %s", source)
        return []
    return await fn(query, max_results)


async def search_many(
    query: str,
    sources: list[str],
    max_per_source: int = 4,
) -> dict[str, list[SearchResult]]:
    """Run searches across multiple sources in parallel."""
    tasks = [search_one(s, query, max_per_source) for s in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: dict[str, list[SearchResult]] = {}
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            logger.warning("source %s failed: %s", source, result)
            out[source] = []
        else:
            out[source] = result
    return out
