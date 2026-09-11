# Eval Harness

Pushes a curated topic set through the pipeline and scores each report
against six metrics — five free, one LLM-based (reused from the Critic).

## Run

```bash
# Full run, LangGraph engine, brief depth, critic on
python -m eval.run

# CrewAI engine
python -m eval.run --engine crew

# Smoke run: first 3 topics
python -m eval.run --limit 3

# Custom topic set
python -m eval.run --topics path/to/topics.yaml --out eval_output/exp-01/

# Cheaper baseline without grounding metric
python -m eval.run --disable-critic
```

Outputs land in `eval_output/<timestamp>/`:

- `summary.md` — human-readable markdown table + aggregate row
- `summary.json` — same metrics in machine-readable form
- `<slug>.json` — per-topic spec + full `ResearchReport` + metrics
- `<slug>.error.txt` — captured exception if a topic crashed

`eval_output/` is gitignored.

## Metrics

| Metric | Cost | What it measures |
|---|---|---|
| `citation_validity` | free | % of `[N]` markers that map to a real source index in `report.sources`. Catches "phantom citations" the LLM invents. |
| `citation_grounding` | LLM | Critic verdict: % of claims whose cited snippet actually supports them. Needs `--enable-critic` (default). |
| `source_diversity` | free | Distinct URL domains / total sources. Higher = drawing from more places, not just one site. |
| `routing_precision` | free | Fraction of a topic's `expected_sources` actually picked by the router. Exercises `app/tools/router.py`. |
| `word_count` | free | Report length. The portfolio claim is 2,000+ words on detailed depth. |
| `elapsed_seconds` | free | Wall-clock per topic. Useful when comparing engines or LLM backends. |

`repaired` indicates the Critic triggered a one-shot Writer repair pass.

## Topic set

`topics.yaml` ships with 10 hand-picked topics covering the five routing
categories — concept-define, paper, code-tools, trend, and mixed — so
every source client gets exercised at least twice. Each entry:

```yaml
- topic: "What is GraphRAG?"
  category: concept-define
  expected_sources: [wikipedia]
  notes: "Definition query; should hit Wikipedia via 'what is'."
```

`expected_sources` is the subset the router *should* pick for that
topic (DuckDuckGo is always included so it's omitted from this list).

## Cost note

Each topic = one full pipeline run = roughly 3–5 LLM calls on the worker
backend, plus the Critic's verdict call (and a repair pass if it
triggers). At ~$0.05–$0.20 per topic depending on backend, a 10-topic
run is $0.50–$2 of API spend. No CI integration for this reason — run
locally before pushing.
