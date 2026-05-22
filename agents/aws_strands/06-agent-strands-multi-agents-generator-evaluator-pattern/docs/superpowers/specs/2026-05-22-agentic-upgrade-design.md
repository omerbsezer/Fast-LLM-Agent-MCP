# Agentic Upgrade Design — Planner + LLM-Router + Dynamic Graph

**Date:** 2026-05-22  
**Project:** 06-agent-strands-multi-agents-generator-evaluator-pattern  
**Status:** Approved

---

## Problem

The current system is an agentic pipeline but not a fully autonomous agent:
- Graph edges are hardcoded Python `if` statements
- Tool selection is pre-assigned per agent (no runtime choice)
- No planner — every run uses the same strategy regardless of topic

## Goal

Upgrade to a genuinely agentic multi-agent system by adding:
1. **Planner agent** — decides strategy for this specific topic before any work begins
2. **LLM-driven tool selection** — agents choose from a shared tool pool at runtime
3. **Dynamic graph** — a Router LLM makes every edge decision, no hardcoded routing

---

## Architecture

### Graph Structure

```
START
  │
  ▼
┌─────────┐   always   ┌──────────┐
│ planner │ ─────────► │  router  │◄──────────────┐
└─────────┘            └────┬─────┘               │
                            │                      │
              ┌─────────────┼─────────────┐        │
              ▼             ▼             ▼        │
        "generate"     "evaluate"      "done"      │
              │             │             │        │
              ▼             ▼             ▼        │
         ┌─────────┐  ┌──────────┐     END        │
         │generator│  │evaluator │                 │
         └────┬────┘  └────┬─────┘                │
              │             │                      │
              └─────────────┴──────────────────────┘
                        always → router
```

### Node Responsibilities

| Node | Runs | Purpose |
|---|---|---|
| `planner` | Once at start | Writes topic strategy to memory: research angle, quality bar, tone, expected iterations |
| `router` | After every node | Reads full memory state, outputs `ROUTING_DECISION: {"next": "generator"\|"evaluator"\|"done", "reason": "..."}` |
| `generator` | 1–N times | Writes/revises the blog post; chooses own tools |
| `evaluator` | 1–N times | Scores and critiques the blog; chooses own tools |

### Edge Conditions

All edges from `router` are decided by parsing the router's text output. Python condition functions extract `ROUTING_DECISION` JSON — no hardcoded logic:

```python
def _route_to_generator(state) -> bool:
    return _parse_routing(state) == "generator"

def _route_to_evaluator(state) -> bool:
    return _parse_routing(state) == "evaluator"

def _route_done(state) -> bool:
    return _parse_routing(state) == "done"
```

---

## Tool Pool

Tools are implemented as `@tool`-decorated functions. Each agent gets a subset — the LLM decides which ones to call and in what order.

| Tool | Signature | Available to |
|---|---|---|
| `web_search` | `(query: str) → str` | Generator, Evaluator |
| `fetch_url` | `(url: str) → str` | Generator, Evaluator |
| `memory_read` | `() → str` | All 4 agents |
| `memory_write` | `(section: str, content: str) → str` | Planner, Generator, Evaluator |
| `summarize_research` | `(text: str) → str` | Generator |
| `fact_check` | `(claim: str) → str` | Evaluator (calls web_search internally) |
| `save_evaluation` | `(verdict, depth_score, recency_score, structure_score, writing_score, feedback) → str` | Evaluator |

**LLM-driven tool selection in practice:**
- Generator may call `web_search` once and write directly, or may call `web_search` → `fetch_url` × N → `summarize_research` before writing
- Evaluator may skip `fact_check` if content looks solid, or call it on suspicious claims
- Neither path is hardcoded — the model decides based on what it encounters

---

## State Management

### Memory File (`output/memory.md`)

Shared state store written by all agents, read by all agents:

```markdown
# Memory — {topic}  ({timestamp})

## Plan
← planner writes strategy (research angle, quality bar, tone, max_iter target)

## Research
← generator writes gathered research per iteration

## Sources
← generator logs article URLs

## Critiques
← evaluator writes scored feedback per iteration

## Route Log
← router logs each ROUTING_DECISION with reason

## Log
← timestamps and iteration counts
```

### Node-to-Node Pass-Through

Strands Graph passes each node's text output as the next node's input:
- Planner output → Router: "Here is my plan, what is the first step?"
- Router output → Generator/Evaluator: Contains routing reason and any specific instructions
- Generator output (blog markdown) → Router: "Blog written, what now?"
- Evaluator output (score summary) → Router: "Evaluation complete, what now?"

### Output Files

- `output/blog_post.md` — saved from `result.results["generator"].result` after graph completes
- `output/memory.md` — written throughout by all agents

---

## Safety & Limits

- `set_max_node_executions(max_iter * 3 + 2)` — prevents infinite router cycles
- Router system prompt instructs it to count generator runs from the `## Log` section of memory and output `"done"` once `max_iter` runs have occurred regardless of verdict
- `set_execution_timeout(600)` — 10-minute hard stop

---

## What Makes This Agentic

| Property | Before | After |
|---|---|---|
| Routing | Python `if` on memory file | LLM Router agent decides every step |
| Tool use | Pre-assigned per agent | Model selects from pool at runtime |
| Strategy | Same every run | Planner writes topic-specific strategy |
| Graph | Fixed edges | Dynamic edges via LLM output parsing |

---

## Files Changed

- `agent.py` — full rewrite with 4 agents, expanded tool pool, dynamic graph
- `requirements.txt` — no new dependencies (strands-agents already covers everything)
