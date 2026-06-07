import re, sys, json, datetime, requests
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()
OUT  = Path("output"); OUT.mkdir(exist_ok=True)
BLOG = OUT / "blog_post.md"
MEM  = OUT / "memory.md"

SITE         = "dev.to"
MIN_ARTICLES = 5
MAX_ITER     = 3

_gen_model     = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.8)
_eval_model    = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.0)
_plan_model    = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.3)
_summary_model = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.0)

_state: dict = {}

def _state_init(topic: str, max_iter: int) -> None:
    _state.update(topic=topic, max_iter=max_iter, iteration=0,
                  blog="", feedback=None, keywords=[], done=False)

# Persistent memory 
def _mem_init(topic: str) -> None:
    MEM.write_text(
        f"# Memory — {topic}  ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n---\n\n"
        "## Research\n\n## Sources\n\n## Critiques\n\n## Log\n",
        encoding="utf-8",
    )

def _mem_read() -> str:
    return MEM.read_text(encoding="utf-8") if MEM.exists() else ""

def _mem_append(section: str, content: str) -> None:
    text = _mem_read()
    eol  = text.index("\n", text.index(f"## {section}") + len(f"## {section}"))
    MEM.write_text(text[:eol+1] + "\n" + content.strip() + "\n" + text[eol+1:], encoding="utf-8")

# Raw fetch (full page for summarizer) 
def _fetch_raw(url: str, snippet: str = "") -> str:
    """Fetch a page and return ALL readable text (no char limit)."""
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        # dev.to: strip nav header that precedes the article body
        if SITE in url:
            m = re.search(r"(?:&nbsp;\s*){3,}", html)
            if m:
                html = html[m.end():]
        text = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text if len(text) >= 300 else (snippet or "[unavailable]")
    except Exception as e:
        return f"[failed: {e}]"

def _ddgs_search(query: str) -> list[dict]:
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

# Summarizer Agent
def _summarize(title: str, url: str, full_text: str, topic: str) -> str:
    """Use an LLM to extract key technical points from a full article."""
    agent = Agent(
        model=_summary_model,
        system_prompt=(
            "You are a research assistant. Given raw article text, extract the most important "
            "technical points relevant to the given topic. Be concise and specific. "
            "Output a bullet-point summary (5-10 bullets). No preamble, no URLs, just bullets."
        ),
    )
    # Cap input to 12 000 chars, enough to cover a full dev.to post
    truncated = full_text[:12_000]
    response  = agent(
        f"Topic we are researching: {topic}\n\n"
        f"Article title: {title}\n"
        f"Article URL: {url}\n\n"
        f"Full article text:\n{truncated}\n\n"
        "Extract the key technical points relevant to the topic."
    )
    return str(response).strip()

def _collect_articles(keywords: list[str], topic: str) -> tuple[str, list[str]]:
    """Search, fetch full pages, summarize each, return (markdown_block, url_list)."""
    seen, found = set(), []
    for kw in keywords:
        if len(found) >= MIN_ARTICLES:
            break
        for hit in _ddgs_search(kw):
            if hit["url"] in seen or len(found) >= MIN_ARTICLES:
                continue
            seen.add(hit["url"])
            raw = _fetch_raw(hit["url"], hit["snippet"])
            if raw.startswith("["):
                continue
            print(f"    ↳ summarising [{len(found)+1}/{MIN_ARTICLES}]  {hit['url']}")
            summary = _summarize(hit["title"], hit["url"], raw, topic)
            if len(summary) < 50:
                continue
            found.append({**hit, "summary": summary})
    if len(found) < MIN_ARTICLES:
        print(f"    ⚠ only {len(found)}/{MIN_ARTICLES} articles summarised")
    sections = [
        f"### [{a['title']}]({a['url']})\n**Snippet:** {a['snippet']}\n\n**Key Points:**\n{a['summary']}\n"
        for a in found
    ]
    return "\n---\n".join(sections) or "_No results._", [a["url"] for a in found]

#  EvalResult schema
class EvalResult(BaseModel):
    verdict:         str
    depth_score:     int = Field(ge=1, le=5)
    recency_score:   int = Field(ge=1, le=5)
    structure_score: int = Field(ge=1, le=5)
    writing_score:   int = Field(ge=1, le=5)
    feedback:        Optional[str] = None

@tool
def search_web(query: str) -> str:
    """Search dev.to for technical articles matching the query.
    Returns a JSON list of {title, url, snippet} objects."""
    return json.dumps(_ddgs_search(query), ensure_ascii=False)

@tool
def fetch_page(url: str) -> str:
    """Fetch and return the full readable text content of a web page."""
    return _fetch_raw(url)

@tool
def summarize_article(url: str, title: str = "") -> str:
    """Fetch a web page and return an LLM-generated bullet-point summary
    of its key technical points relevant to the current pipeline topic.
    Use this instead of fetch_page when you want distilled insight, not raw text."""
    raw = _fetch_raw(url)
    if raw.startswith("["):
        return raw
    return _summarize(title or url, url, raw, _state.get("topic", ""))

@tool
def read_memory() -> str:
    """Read the current persistent memory (research notes, sources, critiques, log)."""
    return _mem_read() or "Memory is empty."

@tool
def write_memory(section: str, content: str) -> str:
    """Append content to a memory section.
    Valid sections: Research, Sources, Critiques, Log."""
    try:
        _mem_append(section, content)
        return f"✓ Written to ## {section}"
    except Exception as e:
        return f"✗ Failed: {e}"

# Generator Agent
@tool
def generate_blog(extra_instructions: str = "") -> str:
    """Run the GeneratorAgent: collect + summarise research, then write/rewrite the blog post."""
    topic     = _state["topic"]
    iteration = _state["iteration"] + 1
    keywords  = _state["keywords"]
    prev_blog = _state["blog"]
    feedback  = _state["feedback"]

    print(f"\n  [GeneratorAgent] writing iter {iteration}/{_state['max_iter']}")

    # Step 1: keyword generation 
    if not keywords:
        kw_agent = Agent(model=_gen_model,
                         system_prompt="Return ONLY a JSON array of strings. No prose, no explanation.")
        kw_text  = str(kw_agent(f"8 diverse search queries for: {topic}")).strip()
        m = re.search(r"\[.*\]", kw_text, re.DOTALL)
        try:    keywords = json.loads(m.group() if m else kw_text)[:8]
        except: keywords = [topic]
        _state["keywords"] = keywords

    # Step 2: collect + summarise 
    research, urls = _collect_articles(keywords, topic)
    now = datetime.datetime.now().strftime("%H:%M:%S")

    # Persist to memory BEFORE calling the writer LLM
    _state["iteration"] = iteration
    _mem_append("Research", f"### Iter {iteration} — {now}\n**KW:** {', '.join(keywords)}\n\n{research}\n")
    _mem_append("Sources",  f"### Iter {iteration}\n" + "\n".join(f"- {u}" for u in urls) + "\n")
    _mem_append("Log",      f"- **Iter {iteration}** `{now}` — {len(urls)} articles\n")

    # Step 3: write with a fresh writer agent
    rewrite_block = ""
    if feedback and iteration > 1:
        rewrite_block = f"\n\nPREVIOUS DRAFT:\n{prev_blog[:3000]}\n\nEVALUATOR FEEDBACK:\n{feedback}"

    memory_ctx = _mem_read().split("## Log")[0][:4000]

    gen_agent = Agent(
        model=_gen_model,
        tools=[search_web, summarize_article, write_memory],
        system_prompt=(
            "You are an expert technical writer for senior AI/ML engineers. "
            "You may call summarize_article(url) to get distilled insights from any URL. "
            "Produce a complete Markdown blog post: # title, ## sections, code examples, "
            "≥8 cited sources, ## References at the end. "
            "Output ONLY the Markdown — no JSON, no prose, no explanations."
        ),
    )

    blog = str(gen_agent(
        f"Write a 1500-2000 word technical blog post about **{topic}** for senior AI/ML engineers.\n\n"
        f"MEMORY CONTEXT:\n{memory_ctx}\n\n"
        f"RESEARCH (pre-summarised key points per article):\n{research}"
        f"{rewrite_block}\n\n"
        f"EXTRA INSTRUCTIONS: {extra_instructions}\n\n"
        "Output ONLY the Markdown blog post. Start directly with the # title."
    )).strip()

    BLOG.write_text(blog, encoding="utf-8")
    _state["blog"] = blog

    print(f"    ↳ blog: {len(blog)} chars | {len(urls)} sources")
    return f"Blog written: {len(blog)} chars, {len(urls)} sources, iteration {iteration}."

# Evaluator Agent 
@tool
def evaluate_blog() -> str:
    """Run the EvaluatorAgent on the current blog draft. Returns JSON: verdict (accepted|rejected), four 1-5 scores, feedback. verdict='accepted' means ALL scores ≥ 4."""
    topic     = _state["topic"]
    iteration = _state["iteration"]

    if not BLOG.exists():
        return json.dumps({"verdict": "rejected", "feedback": "No blog post found. Generate one first."})

    print(f"\n  [EvaluatorAgent] reviewing iter {iteration}")

    eval_agent = Agent(
        model=_eval_model,
        tools=[read_memory],
        system_prompt=(
            "You are a rigorous technical editor scoring blog posts for senior AI/ML engineers. "
            "Use read_memory to check research history and prior critiques before scoring. "
            "Respond ONLY with valid JSON — no markdown fences, no prose."
        ),
    )

    text = str(eval_agent(
        f"Evaluate this blog post on **{topic}** for senior AI engineers.\n\n"
        f"POST:\n{BLOG.read_text(encoding='utf-8')}\n\n"
        "Score 1-5 each: depth, recency (2023-2025), structure, writing. Accept only if ALL ≥ 4.\n\n"
        'Return ONLY: {"verdict":"accepted"|"rejected","depth_score":int,"recency_score":int,'
        '"structure_score":int,"writing_score":int,"feedback":"str or null"}'
    )).strip()

    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    m    = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        e = EvalResult(**(json.loads(m.group() if m else text)))
    except Exception as ex:
        print(f"    ⚠ parse error: {ex}")
        e = EvalResult(verdict="rejected", depth_score=1, recency_score=1,
                       structure_score=1, writing_score=1, feedback=str(ex))

    print(f"    ↳ D:{e.depth_score} R:{e.recency_score} S:{e.structure_score} W:{e.writing_score} → {e.verdict.upper()}")

    now = datetime.datetime.now().strftime("%H:%M:%S")
    _mem_append("Critiques",
        f"### Iter {iteration} — {now} — **{e.verdict.upper()}**\n"
        f"| Depth | Recency | Structure | Writing |\n|---|---|---|---|\n"
        f"| {e.depth_score}/5 | {e.recency_score}/5 | {e.structure_score}/5 | {e.writing_score}/5 |\n"
        + (f"\n**Feedback:** {e.feedback}\n" if e.feedback else "")
    )
    _mem_append("Log",
        f"- **Eval iter {iteration}** `{now}` — {e.verdict.upper()} "
        f"(D:{e.depth_score} R:{e.recency_score} S:{e.structure_score} W:{e.writing_score})\n"
    )
    _state["feedback"] = e.feedback or ""
    if e.verdict == "accepted":
        _state["done"] = True
    return e.model_dump_json()

# Planner Agent 
def run(topic: str, max_iter: int = MAX_ITER) -> None:
    print(f"\n{'='*60}\n  TOPIC : {topic}\n  MAX ITER: {max_iter}\n{'='*60}")
    _state_init(topic, max_iter)
    _mem_init(topic)

    planner = Agent(
        model=_plan_model,
        tools=[generate_blog, evaluate_blog, search_web, fetch_page,
               summarize_article, read_memory, write_memory],
        system_prompt=(
            "You are an autonomous blog-pipeline orchestrator for senior AI/ML engineering content.\n\n"
            "Goal: produce a high-quality technical blog post (all eval scores ≥ 4) "
            f"within the allowed iterations.\n\n"
            "Tools available:\n"
            "  • generate_blog(extra_instructions)  — research + summarise + write / rewrite\n"
            "  • evaluate_blog()                    — score the current draft\n"
            "  • search_web(query)                  — ad-hoc article search\n"
            "  • fetch_page(url)                    — full raw text of a page\n"
            "  • summarize_article(url, title)      — LLM-distilled key points from a page\n"
            "  • read_memory()                      — inspect research & critiques\n"
            "  • write_memory(section, content)     — log planning notes\n\n"
            "Flow:\n"
            "  1. generate_blog() → 2. evaluate_blog() → 3. if rejected, generate_blog(extra_instructions=<fix>) → repeat.\n"
            "  Stop when accepted or iteration budget exhausted.\n"
            "  Use summarize_article between steps when you need deeper insight on a specific source."
        ),
    )

    print("\n[PlannerAgent] starting autonomous pipeline...\n")
    planner(
        f"Topic: **{topic}**\n"
        f"Max iterations allowed: {max_iter}\n\n"
        "Run the full pipeline autonomously. Generate, evaluate, and iterate "
        "until the post is accepted or the budget is exhausted."
    )

    print(f"\n{'='*60}")
    print(f"  Done — iteration {_state['iteration']} | accepted={_state['done']}")
    print(f"  Output: {BLOG}  |  Memory: {MEM}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "AI Agents with Memory on AWS Bedrock AgentCore")