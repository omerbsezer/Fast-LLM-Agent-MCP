import re, sys, datetime, requests
from pathlib import Path
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from ddgs import DDGS
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()
TARGET_SITES = ["builder.aws.com", "dev.to"]

MEMORY = Path("memory.md")
BLOG   = Path("blog_post.md")
GEN    = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.8)
EVAL   = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0)
S_RESEARCH  = "Research Notes"
S_SOURCES   = "Sources Used"
S_CRITIQUES = "Evaluator Critiques"
S_LOG       = "Iteration Log"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

# Search with DDGS
def is_target_url(url: str) -> bool:
    """Strictly allow only URLs from TARGET_SITES — reject ads, bing, redirects."""
    return any(f"://{site}" in url or f".{site}" in url for site in TARGET_SITES)

def search_site(query: str, site: str, max_results: int = 3) -> list[dict]:
    """DuckDuckGo search; tries site: operator, falls back to plain query filtered by URL."""
    def _run(q: str) -> list[dict]:
        with DDGS() as ddgs:
            raw = list(ddgs.text(q, max_results=max_results * 4))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw if is_target_url(r.get("href", ""))
        ]

    for q in [f"site:{site} {query}", f"{query} {site}"]:
        try:
            results = _run(q)
            if results:
                return results[:max_results]
        except Exception:
            continue
    return []

def fetch_article(url: str, max_chars: int = 2000) -> str:
    """Fetch URL, strip HTML, return plain text excerpt."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        text = re.sub(r"<style[^>]*>.*?</style>", " ", resp.text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"[fetch failed: {e}]"

def research(query: str) -> tuple[str, list[str]]:
    """Search each TARGET_SITE, fetch articles, return (formatted_text, real_urls)."""
    sections, real_urls = [], []
    for site in TARGET_SITES:
        hits = search_site(query, site, max_results=2)
        if not hits:
            print(f"    [{site}] No clean results for: {query!r}")
            continue
        for hit in hits:
            content = fetch_article(hit["url"])
            sections.append(
                f"### [{hit['title']}]({hit['url']})\n"
                f"**Source:** {site}\n"
                f"**Snippet:** {hit['snippet']}\n\n"
                f"**Content:**\n{content}\n"
            )
            real_urls.append(hit["url"])
            print(f"    [{site}] OK: {hit['url']}")

    text = "\n---\n".join(sections) if sections else f"_No results from {TARGET_SITES}._"
    return text, real_urls

# Memory 
def mem_read() -> str:
    return MEMORY.read_text(encoding="utf-8") if MEMORY.exists() else ""

def mem_append(section: str, content: str) -> None:
    text, marker = mem_read(), f"## {section}"
    if marker not in text:
        raise ValueError(f"Section '{marker}' not found in memory.md")
    eol = text.index("\n", text.index(marker) + len(marker))
    MEMORY.write_text(text[:eol+1] + "\n" + content.strip() + "\n" + text[eol+1:], encoding="utf-8")

def mem_init(topic: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    MEMORY.write_text(
        f"# Shared Agent Memory\n**Topic:** {topic}\n**Started:** {now}\n\n---\n\n"
        f"## {S_RESEARCH}\n\n## {S_SOURCES}\n\n## {S_CRITIQUES}\n\n## {S_LOG}\n",
        encoding="utf-8"
    )

class State(TypedDict):
    topic: str; blog: str; feedback: Optional[str]
    verdict: str; iteration: int; max_iter: int

class Eval(BaseModel):
    verdict:         Literal["accepted", "rejected"]
    depth_score:     int = Field(ge=1, le=5)
    recency_score:   int = Field(ge=1, le=5)
    structure_score: int = Field(ge=1, le=5)
    writing_score:   int = Field(ge=1, le=5)
    feedback:        Optional[str] = None

evaluator = EVAL.with_structured_output(Eval)

def generator_node(state: State) -> dict:
    n, topic = state["iteration"] + 1, state["topic"]
    print(f"\n  [Generator] Iter {n}/{state['max_iter']}")

    all_research, all_urls = [], []
    for q in [topic, f"{topic} tutorial", f"{topic} best practices 2025"]:
        print(f"    Searching: {q!r}")
        text, urls = research(q)
        all_research.append(f"## Query: `{q}`\n\n{text}")
        all_urls.extend(urls)

    combined   = "\n\n".join(all_research)
    real_urls  = list(dict.fromkeys(all_urls))  # deduplicate

    now = datetime.datetime.now().strftime("%H:%M:%S")
    mem_append(S_RESEARCH,
        f"### Iter {n} — {now}\n**Sites:** {', '.join(TARGET_SITES)}\n\n{combined}\n"
    )
    mem_append(S_SOURCES,
        f"### Iter {n}\n" + "\n".join(f"- {u}" for u in real_urls) + "\n"
    )
    mem_append(S_LOG, f"- **Iter {n}** `{now}` — fetched {len(real_urls)} articles from target sites.\n")

    rewrite_ctx = (
        f"\n\nPREVIOUS POST:\n{state['blog']}\n\nFEEDBACK:\n{state['feedback']}"
        if state.get("feedback") and n > 1 else ""
    )
    prompt = (
        f"You are an expert AI engineering blogger. Write a detailed blog post (700-1000 words) about **{topic}**.\n\n"
        f"Use ONLY the research below. Cite real article titles and their exact URLs.\n\n"
        f"{combined[:4000]}"
        f"{rewrite_ctx}\n\n"
        "Format: compelling # title, ## sections, code examples where relevant.\n"
        "End with ## References listing ONLY the real URLs from the research above.\n"
        "IMPORTANT: Do not invent or hallucinate any URLs. Only use URLs that appear in the research.\n"
        "Output Markdown only."
    )
    blog = GEN.invoke(prompt).content
    BLOG.write_text(blog, encoding="utf-8")
    print(f"    Blog written ({len(blog)} chars), {len(real_urls)} verified sources")
    return {"blog": blog, "iteration": n}


def evaluator_node(state: State) -> dict:
    print(f"  [Evaluator] Reviewing iter {state['iteration']}...")
    e: Eval = evaluator.invoke(
        f"Evaluate this blog post on **{state['topic']}** for senior AI engineers.\n\n"
        f"POST:\n{BLOG.read_text(encoding='utf-8')}\n\nMEMORY:\n{mem_read()[:1500]}\n\n"
        "Score 1-5 each (depth, recency 2023-2025, structure, writing). Accept only if ALL >= 4."
    )
    print(f"  D:{e.depth_score} R:{e.recency_score} S:{e.structure_score} W:{e.writing_score} -> {e.verdict.upper()}")

    now = datetime.datetime.now().strftime("%H:%M:%S")
    mem_append(S_CRITIQUES,
        f"### Iter {state['iteration']} — {now} — **{e.verdict.upper()}**\n"
        f"| Criterion | Score |\n|---|---|\n"
        f"| Depth | {e.depth_score}/5 |\n| Recency | {e.recency_score}/5 |\n"
        f"| Structure | {e.structure_score}/5 |\n| Writing | {e.writing_score}/5 |\n\n"
        + (f"**Feedback:** {e.feedback}\n" if e.feedback else "")
    )
    mem_append(S_LOG,
        f"- **Iter {state['iteration']}** `{now}` — Evaluator: {e.verdict.upper()} "
        f"(D:{e.depth_score} R:{e.recency_score} S:{e.structure_score} W:{e.writing_score})\n"
    )
    return {"verdict": e.verdict, "feedback": e.feedback or ""}


def route(state: State) -> str:
    if state["verdict"] == "accepted" or state["iteration"] >= state["max_iter"]:
        if state["iteration"] >= state["max_iter"]: print("  Max iterations reached — publishing best version.")
        return "Accepted"
    return "Rejected"

workflow = StateGraph(State)
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "evaluator")
workflow.add_conditional_edges("evaluator", route, {"Accepted": END, "Rejected": "generator"})
pipeline = workflow.compile()

def run(topic: str, max_iter: int = 3) -> None:
    print(f"\n{'='*60}\n  TOPIC: {topic}\n{'-'*60}")
    mem_init(topic)
    result = pipeline.invoke({"topic": topic, "blog": "", "feedback": None, "verdict": "", "iteration": 0, "max_iter": max_iter})
    print(f"\n{'-'*60}\n  Done in {result['iteration']} iteration(s) | blog_post.md + memory.md written\n{'='*60}\n")
    print(result["blog"])

if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "How AI Agents")