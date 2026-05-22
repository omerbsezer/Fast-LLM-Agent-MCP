import re, sys, json, datetime, requests, boto3
from pathlib import Path
from strands import Agent, tool
from strands.multiagent import GraphBuilder
from strands.models import BedrockModel
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

OUT  = Path("output"); OUT.mkdir(exist_ok=True)
BLOG = OUT / "blog_post.md"
MEM  = OUT / "memory.md"

SITE         = "dev.to"
MIN_ARTICLES = 5
REGION       = "us-east-1"
MODEL_ID     = "us.amazon.nova-pro-v1:0"

gen_model    = BedrockModel(model_id=MODEL_ID, temperature=0.8)
eval_model   = BedrockModel(model_id=MODEL_ID, temperature=0)
router_model = BedrockModel(model_id=MODEL_ID, temperature=0)
plan_model   = BedrockModel(model_id=MODEL_ID, temperature=0.3)

# ── Memory ─────────────────────────────────────────────────────────────────────
def mem_init(topic: str) -> None:
    MEM.write_text(
        f"# Memory — {topic}  ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n---\n\n"
        "## Plan\n\n## Research\n\n## Sources\n\n## Critiques\n\n## Route Log\n\n## Log\n",
        encoding="utf-8",
    )

def mem_read() -> str:
    return MEM.read_text(encoding="utf-8") if MEM.exists() else ""

def mem_append(section: str, content: str) -> None:
    text = mem_read()
    eol  = text.index("\n", text.index(f"## {section}") + len(f"## {section}"))
    MEM.write_text(text[:eol+1] + "\n" + content.strip() + "\n" + text[eol+1:], encoding="utf-8")

def mem_count_generator_runs() -> int:
    mem = mem_read()
    if "## Log" not in mem:
        return 0
    return mem.split("## Log")[1].count("generator_run")

# ── Research helpers ────────────────────────────────────────────────────────────
def _search(query: str) -> list[dict]:
    def run(q: str) -> list[dict]:
        with DDGS() as d:
            raw = list(d.text(q, max_results=MIN_ARTICLES * 4))
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in raw if SITE in r.get("href", "")]
    for q in [f"site:{SITE} {query}", f"{query} {SITE}"]:
        try:
            hits = run(q)
            if hits:
                return hits[:MIN_ARTICLES]
        except Exception:
            continue
    return []

def _fetch(url: str, snippet: str = "") -> str:
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        text = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:3000] if len(text) >= 300 else (snippet[:3000] or "[unavailable]")
    except Exception as e:
        return f"[failed: {e}]"

# ── Tools ───────────────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search for web articles matching the query. Returns JSON list of {title, url, snippet}."""
    results = _search(query)
    print(f"  [web_search] '{query}' → {len(results)} results")
    return json.dumps(results)

@tool
def fetch_url(url: str) -> str:
    """Fetch and extract readable text content from a URL."""
    content = _fetch(url)
    print(f"  [fetch_url] {url[:60]}... → {len(content)} chars")
    return content

@tool
def memory_read() -> str:
    """Read the full shared memory context (plan, research, sources, critiques, route log)."""
    return mem_read()

@tool
def memory_write(section: str, content: str) -> str:
    """Append content to a section in shared memory. section: Plan, Research, Sources, Critiques, Route Log, or Log."""
    mem_append(section, content)
    return f"Written to ## {section}"

@tool
def summarize_research(text: str) -> str:
    """Condense a block of research text into key technical bullet points. Use before writing."""
    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)
        resp   = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text":
                f"Summarize the following research into 5-8 key technical bullet points "
                f"for a blog post author:\n\n{text[:6000]}"
            }]}],
            inferenceConfig={"temperature": 0.3},
        )
        summary = resp["output"]["message"]["content"][0]["text"]
        print(f"  [summarize_research] {len(text)} → {len(summary)} chars")
        return summary
    except Exception as e:
        return f"[summarize failed: {e}]\n\nOriginal (truncated):\n{text[:2000]}"

@tool
def fact_check(claim: str) -> str:
    """Search for evidence supporting or contradicting a specific technical claim."""
    print(f"  [fact_check] '{claim[:80]}'")
    results = _search(claim + " 2024 2025")
    if not results:
        return f"No evidence found for: {claim}"
    snippets = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results[:3])
    return f"Evidence for '{claim}':\n{snippets}"

@tool
def save_evaluation(verdict: str, depth_score: int, recency_score: int,
                    structure_score: int, writing_score: int, feedback: str) -> str:
    """Save evaluation to memory. verdict: 'accepted' or 'rejected'."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    mem_append("Critiques",
        f"### {now} — **{verdict.upper()}**\n"
        f"| Depth | Recency | Structure | Writing |\n|---|---|---|---|\n"
        f"| {depth_score}/5 | {recency_score}/5 | {structure_score}/5 | {writing_score}/5 |\n"
        + (f"\n**Feedback:** {feedback}\n" if feedback else "")
    )
    mem_append("Log",
        f"- `{now}` — EVAL {verdict.upper()} "
        f"(D:{depth_score} R:{recency_score} S:{structure_score} W:{writing_score})\n"
    )
    print(f"  [evaluator] D:{depth_score} R:{recency_score} S:{structure_score} W:{writing_score} → {verdict.upper()}")
    return verdict

# ── Routing ─────────────────────────────────────────────────────────────────────
def _parse_routing(state) -> str:
    """Extract next-step from router node's ROUTING_DECISION JSON token."""
    router_result = state.results.get("router")
    if not router_result:
        return "generator"
    text = str(router_result.result)
    match = re.search(r'ROUTING_DECISION:\s*(\{[^}]+\})', text)
    if match:
        try:
            return json.loads(match.group(1)).get("next", "done")
        except json.JSONDecodeError:
            pass
    text_lower = text.lower()
    if '"next": "generator"' in text_lower:
        return "generator"
    if '"next": "evaluator"' in text_lower:
        return "evaluator"
    return "done"

def _route_to_generator(state) -> bool:
    return _parse_routing(state) == "generator"

def _route_to_evaluator(state) -> bool:
    return _parse_routing(state) == "evaluator"

# ── Agents ───────────────────────────────────────────────────────────────────────
planner_agent = Agent(
    model=plan_model,
    system_prompt=(
        "You are a strategic planner for technical blog post creation.\n\n"
        "When given a topic and max_iterations:\n"
        "1. Call memory_read() to check if a plan already exists.\n"
        "2. Write a concise strategy covering:\n"
        "   - Research angle: what specific aspects to focus on\n"
        "   - Quality bar: what makes an excellent post on this topic\n"
        "   - Tone and target audience\n"
        "   - Max iterations target\n"
        "3. Call memory_write('Plan', <your strategy>) to persist it.\n\n"
        "Output a brief confirmation after saving."
    ),
    tools=[memory_read, memory_write],
)

router_agent = Agent(
    model=router_model,
    system_prompt=(
        "You are a routing agent deciding the next step in a blog creation workflow.\n\n"
        "1. Call memory_read() to review current state.\n"
        "2. Count 'generator_run' entries in ## Log to know iteration count.\n"
        "3. Decide next action:\n"
        "   - 'generator': no blog written yet, OR last eval was REJECTED and iterations < max_iter\n"
        "   - 'evaluator': blog was just written/revised and has not been evaluated yet\n"
        "   - 'done': last eval was ACCEPTED, OR iteration count reached max_iter\n"
        "4. Log your decision: call memory_write('Route Log', "
        "'- `HH:MM:SS` → <next>: <reason>\\n').\n"
        "5. End your response with EXACTLY this line (required for routing):\n"
        "   ROUTING_DECISION: {\"next\": \"generator\"|\"evaluator\"|\"done\", \"reason\": \"<one line>\"}"
    ),
    tools=[memory_read, memory_write],
)

generator_agent = Agent(
    model=gen_model,
    system_prompt=(
        "You are a technical blog post generator for senior AI/ML engineers.\n\n"
        "Available tools — pick what you need:\n"
        "- web_search(query): search dev.to for articles (returns JSON)\n"
        "- fetch_url(url): get full text of a specific article\n"
        "- summarize_research(text): condense gathered text into bullet points\n"
        "- memory_read(): review the plan and prior evaluator feedback\n"
        "- memory_write(section, content): save research/sources/log entries\n\n"
        "Steps:\n"
        "1. Call memory_read() to get the plan and any prior critique\n"
        "2. If ## Research in memory is empty: search and fetch articles, save with "
        "   memory_write('Research', ...) and memory_write('Sources', ...)\n"
        "3. Optionally call summarize_research() on gathered text\n"
        "4. Write blog post; address any critique from ## Critiques\n"
        "5. Call memory_write('Log', '- `HH:MM:SS` generator_run\\n') to record this run\n"
        "6. Output ONLY the markdown blog post starting directly with # Title\n\n"
        "Blog requirements: 1500-2000 words, ≥8 cited sources with URLs, # title, "
        "## sections, code examples, ## References at end."
    ),
    tools=[web_search, fetch_url, summarize_research, memory_read, memory_write],
)

evaluator_agent = Agent(
    model=eval_model,
    system_prompt=(
        "You are a strict technical blog post evaluator for senior AI/ML engineers.\n\n"
        "The blog post to evaluate is provided as your input.\n\n"
        "Available tools — pick what you need:\n"
        "- memory_read(): read the plan's quality bar and prior critiques\n"
        "- fact_check(claim): verify a suspicious technical claim\n"
        "- web_search(query): verify that cited sources exist and are current\n"
        "- save_evaluation(...): record scores and verdict\n\n"
        "Steps:\n"
        "1. Call memory_read() to check the plan's quality bar and prior critiques\n"
        "2. Optionally call fact_check() on suspicious claims\n"
        "3. Score 1-5: depth (technical insight), recency (2023-2025), "
        "   structure (headings/flow), writing (clarity)\n"
        "4. Accept ONLY if ALL four scores ≥ 4\n"
        "5. Call save_evaluation() with scores, verdict, and specific actionable feedback"
    ),
    tools=[memory_read, fact_check, web_search, save_evaluation],
)

# ── Graph ─────────────────────────────────────────────────────────────────────
def build_pipeline(max_iter: int):
    builder = GraphBuilder()
    builder.add_node(planner_agent,   "planner")
    builder.add_node(router_agent,    "router")
    builder.add_node(generator_agent, "generator")
    builder.add_node(evaluator_agent, "evaluator")

    # Fixed edges: always fire after each node
    builder.add_edge("planner",   "router")
    builder.add_edge("generator", "router")
    builder.add_edge("evaluator", "router")

    # Dynamic edges: only traversed when router LLM says so
    # When neither fires (router says "done"), graph terminates naturally
    builder.add_edge("router", "generator", condition=_route_to_generator)
    builder.add_edge("router", "evaluator", condition=_route_to_evaluator)

    builder.set_entry_point("planner")
    builder.set_max_node_executions(max_iter * 3 + 4)
    builder.set_execution_timeout(600)
    builder.reset_on_revisit(True)
    return builder.build()

# ── Entry Point ───────────────────────────────────────────────────────────────
def run(topic: str, max_iter: int = 3) -> None:
    print(f"\n{'='*60}\n  {topic}\n{'='*60}")
    mem_init(topic)
    pipeline = build_pipeline(max_iter)
    result   = pipeline(
        f"Topic: {topic}\nMax iterations: {max_iter}\n"
        "Create a comprehensive technical blog post on this topic."
    )

    gen_node = result.results.get("generator")
    if gen_node and hasattr(gen_node, "result"):
        blog_text = str(gen_node.result)
        BLOG.write_text(blog_text, encoding="utf-8")
        print(f"\n  Blog saved: {len(blog_text)} chars → {BLOG}")

    print(f"\n{'='*60}")
    print(f"  Done — status: {getattr(result, 'status', 'complete')}")
    print(f"  output/ written")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "AI Agents with Memory on AWS Bedrock AgentCore")
