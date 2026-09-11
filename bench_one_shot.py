"""One-shot wall-clock benchmark of the multi-agent research pipeline.

Usage:
    python bench_one_shot.py "RAG evaluation: faithfulness vs answer relevance"

Prints the elapsed seconds. Writes the generated markdown report to
./bench_output/<topic-slug>.md so you can sanity-check quality.

Honest measurement caveats:
- This is wall-clock for ONE topic ONE run. Network variability matters.
- Manual baseline must be measured separately (you doing the same task by hand).
- The 60% / 2x / 3x claim should come from manual_seconds / pipeline_seconds.
"""
import asyncio
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Late import after .env is loaded so LLM_BACKEND env var is honored
from app.graph.pipeline import run_pipeline  # noqa: E402


def _slugify(s: str) -> str:
    return re.sub(r"[^\w\-]+", "-", s.lower()).strip("-")[:60]


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bench_one_shot.py \"<topic>\"")
        sys.exit(1)

    topic = sys.argv[1]
    print(f"Topic: {topic!r}")
    print(f"LLM_BACKEND: {os.getenv('LLM_BACKEND', 'glm')}")
    print(f"Starting pipeline...\n")

    start = time.perf_counter()
    # depth is a string the planner reads in its prompt: "brief" or "detailed".
    # Passing an int silently produced a meaningless "Depth: 3" — the pipeline
    # still ran, but the depth control did nothing.
    report = await run_pipeline(topic, depth="detailed", use_opus_planner=False)
    elapsed = time.perf_counter() - start

    out_dir = Path("bench_output")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / f"{_slugify(topic)}.md"

    # Compose the full markdown including sources at the end so it's a
    # complete artifact (the report.full_report only has inline [1][2][3]
    # citation markers; the source list lives separately on the schema).
    sources_block = "\n\n---\n\n## Sources\n\n"
    if report.sources:
        for i, src in enumerate(report.sources, 1):
            title = getattr(src, "title", "")
            url = getattr(src, "url", "")
            sources_block += f"{i}. [{title}]({url})\n"
    else:
        sources_block += "_(none cited)_\n"

    report_path.write_text(report.full_report + sources_block, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"PIPELINE WALL-CLOCK: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Sources cited: {len(report.sources)}")
    print(f"Report words: {len(report.full_report.split())}")
    print(f"Report saved to: {report_path}")
    print()
    print("Sources:")
    for i, src in enumerate(report.sources, 1):
        print(f"  [{i}] {getattr(src, 'title', '')[:60]}")
        print(f"      {getattr(src, 'url', '')}")
    print(f"{'='*60}")

    # Format something we can paste straight into the cheat sheet:
    print(f"\nFor the cheat sheet:")
    print(f"  Pipeline: {elapsed/60:.1f} min, {len(report.sources)} sources cited.")
    print(f"  Compare against your manual time on the same topic to compute the ratio.")


if __name__ == "__main__":
    asyncio.run(main())
