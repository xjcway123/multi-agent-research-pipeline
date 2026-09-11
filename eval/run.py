"""Eval harness — run the pipeline against a curated topic set and score.

Usage:
    python -m eval.run                              # defaults: langgraph, eval/topics.yaml
    python -m eval.run --engine crew
    python -m eval.run --topics path/to/other.yaml --out eval_output/run-X/
    python -m eval.run --limit 3                    # smoke run: first 3 topics only

Each topic produces a JSON file with the full report + metrics; an
aggregate `summary.md` is written at the end. No-op on cost-sensitive
runs because each topic = one full pipeline call (~$0.05-0.20 depending
on backend and depth).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from eval.metrics import TopicResult, TopicSpec, aggregate, evaluate  # noqa: E402

logger = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _slugify(s: str) -> str:
    return re.sub(r"[^\w\-]+", "-", s.lower()).strip("-")[:60]


def load_topics(path: Path) -> list[TopicSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[TopicSpec] = []
    for entry in raw.get("topics", []):
        out.append(
            TopicSpec(
                topic=entry["topic"],
                category=entry.get("category", "uncategorized"),
                expected_sources=entry.get("expected_sources", []) or [],
                notes=entry.get("notes", ""),
            )
        )
    return out


async def _run_one(
    spec: TopicSpec,
    engine: str,
    depth: str,
    enable_critic: bool,
):
    if engine == "crew":
        from app.crew.pipeline import run_crew_pipeline

        runner = run_crew_pipeline
    else:
        from app.graph.pipeline import run_pipeline

        runner = run_pipeline

    start = time.perf_counter()
    report = await runner(
        spec.topic,
        depth=depth,
        use_opus_planner=False,
        enable_critic=enable_critic,
    )
    elapsed = time.perf_counter() - start
    return report, elapsed


def _summary_table(results: list[TopicResult]) -> str:
    headers = [
        "topic",
        "category",
        "valid",
        "grounded",
        "diversity",
        "routing",
        "words",
        "secs",
        "repaired",
    ]
    rows = [headers]
    for r in results:
        row = r.as_row()
        rows.append([
            row["topic"][:50],
            row["category"],
            f"{row['citation_validity']:.2f}",
            ("-" if row["citation_grounding"] is None else f"{row['citation_grounding']:.2f}"),
            f"{row['source_diversity']:.2f}",
            f"{row['routing_precision']:.2f}",
            str(row["word_count"]),
            str(row["elapsed_seconds"]),
            "yes" if row["repaired"] else "",
        ])
    # Markdown table
    out_lines = ["| " + " | ".join(rows[0]) + " |"]
    out_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows[1:]:
        out_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(out_lines)


def _summary_markdown(
    engine: str,
    depth: str,
    enable_critic: bool,
    results: list[TopicResult],
    agg: dict,
    started_at: str,
) -> str:
    lines = [
        f"# Eval run — {started_at}",
        "",
        f"- engine: `{engine}`",
        f"- depth: `{depth}`",
        f"- critic: `{enable_critic}`",
        f"- topics: {len(results)}",
        "",
        "## Per-topic results",
        "",
        _summary_table(results),
        "",
        "## Aggregate",
        "",
    ]
    for k, v in agg.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.3f}")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["langgraph", "crew"], default="langgraph")
    ap.add_argument("--depth", choices=["brief", "detailed"], default="brief")
    ap.add_argument("--topics", type=Path, default=Path("eval/topics.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--enable-critic", dest="enable_critic", action="store_true", default=True)
    ap.add_argument("--disable-critic", dest="enable_critic", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N topics")
    args = ap.parse_args()

    if not args.topics.exists():
        logger.error("topics file not found: %s", args.topics)
        return 2

    specs = load_topics(args.topics)
    if args.limit > 0:
        specs = specs[: args.limit]
    if not specs:
        logger.error("no topics to run")
        return 2

    started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out or Path(f"eval_output/{started_at}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("writing to %s", out_dir)

    results: list[TopicResult] = []
    for i, spec in enumerate(specs, 1):
        logger.info("[%d/%d] %s", i, len(specs), spec.topic)
        try:
            report, elapsed = await _run_one(
                spec, args.engine, args.depth, args.enable_critic
            )
        except Exception as exc:
            logger.exception("topic failed: %s", spec.topic)
            (out_dir / f"{_slugify(spec.topic)}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            continue

        result = evaluate(spec, report, elapsed)
        results.append(result)

        # Persist the full report + metrics so we can audit later
        (out_dir / f"{_slugify(spec.topic)}.json").write_text(
            json.dumps(
                {
                    "spec": spec.__dict__,
                    "metrics": result.as_row(),
                    "report": report.model_dump(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "  valid=%.2f grounded=%s words=%d secs=%.1f%s",
            result.citation_validity,
            f"{result.citation_grounding:.2f}" if result.citation_grounding is not None else "-",
            result.word_count,
            result.elapsed_seconds,
            " (repaired)" if result.repaired else "",
        )

    agg = aggregate(results)
    summary = _summary_markdown(
        args.engine, args.depth, args.enable_critic, results, agg, started_at
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"config": vars(args) | {"topics": str(args.topics), "out": str(out_dir)},
             "aggregate": agg,
             "results": [r.as_row() for r in results]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
