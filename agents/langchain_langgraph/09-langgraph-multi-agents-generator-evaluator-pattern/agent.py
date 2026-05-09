import re, sys, json, datetime, requests
from pathlib import Path
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()
OUT  = Path("output"); OUT.mkdir(exist_ok=True)
BLOG = OUT / "blog_post.md"
MEM  = OUT / "memory.md"
SITE         = "dev.to"
MIN_ARTICLES = 5
GEN  = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.8)
EVAL = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0)

# Memory 
def mem_init(topic: str) -> None:
    MEM.write_text(
        f"# Memory — {topic}  ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n---\n\n"
        "## Research\n\n## Sources\n\n## Critiques\n\n## Log\n",
        encoding="utf-8",
    )

def mem_read() -> str:
    return MEM.read_text(encoding="utf-8") if MEM.exists() else ""

def mem_append(section: str, content: str) -> None:
    text = mem_read()
    eol  = text.index("\n", text.index(f"## {section}") + len(f"## {section}"))
    MEM.write_text(text[:eol+1] + "\n" + content.strip() + "\n" + text[eol+1:], encoding="utf-8")

# Research
def get_keywords(topic: str) -> list[str]:
    resp = GEN.invoke(
        f"Generate 8 diverse search queries for technical articles about: {topic}\n"
        "Return ONLY a JSON array of strings."
    )
    try:    return json.loads(resp.content.strip())[:8]
    except: return [topic]

def search(query: str) -> list[dict]:
    def run(q: str) -> list[dict]:
        with DDGS() as d:
            raw = list(d.text(q, max_results=MIN_ARTICLES * 4))
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in raw if SITE in r.get("href", "")]
    for q in [f"site:{SITE} {query}", f"{query} {SITE}"]:
        try:
            hits = run(q)
            if hits: return hits[:MIN_ARTICLES]
        except: continue
    return []

def fetch(url: str, snippet: str = "") -> str:
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        text = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:3000] if len(text) >= 300 else (snippet[:3000] or "[unavailable]")
    except Exception as e:
        return f"[failed: {e}]"

def collect(keywords: list[str]) -> tuple[str, list[str]]:
    seen, found = set(), []
    for kw in keywords:
        if len(found) >= MIN_ARTICLES: break
        for hit in search(kw):
            if hit["url"] in seen or len(found) >= MIN_ARTICLES: continue
            seen.add(hit["url"])
            content = fetch(hit["url"], hit["snippet"])
            if len(content.strip()) < 300 or content.startswith("["): continue
            found.append({**hit, "content": content})
            print(f"  {len(found)}/{MIN_ARTICLES}  {hit['url']}")
    if len(found) < MIN_ARTICLES:
        print(f"  WARNING: only {len(found)}/{MIN_ARTICLES} articles found")
    sections = [
        f"### [{a['title']}]({a['url']})\n**Snippet:** {a['snippet']}\n\n**Content:**\n{a['content']}\n"
        for a in found
    ]
    return "\n---\n".join(sections) or "_No results._", [a["url"] for a in found]

# Agents
class State(TypedDict):
    topic: str; keywords: list[str]; blog: str
    feedback: Optional[str]; verdict: str; iteration: int; max_iter: int

class EvalResult(BaseModel):
    verdict:         Literal["accepted", "rejected"]
    depth_score:     int = Field(ge=1, le=5)
    recency_score:   int = Field(ge=1, le=5)
    structure_score: int = Field(ge=1, le=5)
    writing_score:   int = Field(ge=1, le=5)
    feedback:        Optional[str] = None
evaluator = EVAL.with_structured_output(EvalResult)

def generator(state: State) -> dict:
    n, topic = state["iteration"] + 1, state["topic"]
    print(f"\n[Generator] Iter {n}/{state['max_iter']}")

    keywords = state["keywords"] or get_keywords(topic)
    research, urls = collect(keywords)
    now = datetime.datetime.now().strftime("%H:%M:%S")

    mem_append("Research", f"### Iter {n} — {now}\n**Keywords:** {', '.join(keywords)}\n\n{research}\n")
    mem_append("Sources",  f"### Iter {n}\n" + "\n".join(f"- {u}" for u in urls) + "\n")
    mem_append("Log",      f"- **Iter {n}** `{now}` — {len(urls)} articles\n")

    rewrite = (f"\n\nPREVIOUS POST:\n{state['blog']}\nFEEDBACK:\n{state['feedback']}"
               if state.get("feedback") and n > 1 else "")

    blog = GEN.invoke(
        f"Write a 1500-2000 word blog post about **{topic}** for senior AI/ML engineers.\n"
        f"Cite ≥8 of the {len(urls)} sources below by title and URL.\n\n"
        f"Research:\n{research[:8000]}{rewrite}\n\n"
        "Format: # title, ## sections, code examples. End with ## References (real URLs only). Markdown only."
    ).content

    BLOG.write_text(blog, encoding="utf-8")
    print(f"  Blog: {len(blog)} chars | {len(urls)} sources")
    return {"blog": blog, "iteration": n, "keywords": keywords}


def evaluate(state: State) -> dict:
    print(f"\n[Evaluator] Reviewing iter {state['iteration']}...")
    e: EvalResult = evaluator.invoke(
        f"Evaluate this blog post on **{state['topic']}** for senior AI engineers.\n\n"
        f"POST:\n{BLOG.read_text(encoding='utf-8')}\n\nMEMORY:\n{mem_read()[:2000]}\n\n"
        "Score 1-5 each: depth, recency (2023-2025), structure, writing. Accept only if ALL ≥ 4."
    )
    print(f"  D:{e.depth_score} R:{e.recency_score} S:{e.structure_score} W:{e.writing_score} → {e.verdict.upper()}")

    now = datetime.datetime.now().strftime("%H:%M:%S")
    mem_append("Critiques",
        f"### Iter {state['iteration']} — {now} — **{e.verdict.upper()}**\n"
        f"| Depth | Recency | Structure | Writing |\n|---|---|---|---|\n"
        f"| {e.depth_score}/5 | {e.recency_score}/5 | {e.structure_score}/5 | {e.writing_score}/5 |\n"
        + (f"\n**Feedback:** {e.feedback}\n" if e.feedback else "")
    )
    mem_append("Log",
        f"- **Iter {state['iteration']}** `{now}` — {e.verdict.upper()} "
        f"(D:{e.depth_score} R:{e.recency_score} S:{e.structure_score} W:{e.writing_score})\n"
    )
    return {"verdict": e.verdict, "feedback": e.feedback or ""}

def route(state: State) -> str:
    if state["iteration"] >= state["max_iter"]:
        print("  Max iterations — publishing best version."); return "Accepted"
    return "Accepted" if state["verdict"] == "accepted" else "Rejected"

# Graph
graph = StateGraph(State)
graph.add_node("generator", generator)
graph.add_node("evaluator", evaluate)
graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges("evaluator", route, {"Accepted": END, "Rejected": "generator"})
pipeline = graph.compile()

def run(topic: str, max_iter: int = 3) -> None:
    print(f"\n{'='*60}\n  {topic}\n{'='*60}")
    mem_init(topic)
    result = pipeline.invoke({
        "topic": topic, "keywords": [], "blog": "", "feedback": None,
        "verdict": "", "iteration": 0, "max_iter": max_iter,
    })
    print(f"\n{'='*60}\n  Done in {result['iteration']} iter(s) — output/ written\n{'='*60}\n")

if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "AI Agents with Memory on AWS Bedrock AgentCore")
