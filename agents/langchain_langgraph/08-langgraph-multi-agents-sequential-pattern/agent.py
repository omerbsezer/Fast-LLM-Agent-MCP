import re
from typing import TypedDict, Optional
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

"""
A multi-stage LangGraph workflow that writes a blog post through
a series of sequential steps, with quality gates that can loop back.

Pipeline:
  START
    - outline_node       (generate structure)
          -  [gate: is outline detailed enough?]
                - Fail → refine_outline_node → draft_node
                -  Pass → draft_node
                      -  draft_node
                            -  [gate: is draft long enough?]
                                  -  Fail → expand_node → seo_node
                                  -  Pass → seo_node
                                              - END
State tracks every stage so you can inspect the full pipeline output.
"""


llm = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.7)

class State(TypedDict):
    topic:           str
    audience:        str
    outline:         str
    refined_outline: Optional[str]
    draft:           str
    expanded_draft:  Optional[str]
    final_post:      str
    seo_tips:        str
    stages_run:      list[str]   # audit trail

def outline_node(state: State) -> dict:
    """Stage 1 — Generate a blog post outline."""
    print("[1/5] Generating outline...")
    response = llm.invoke(
        f"Create a blog post outline for the topic: '{state['topic']}'\n"
        f"Target audience: {state['audience']}\n"
        f"Include: title, intro hook, 3-5 main sections with bullet points, conclusion.\n"
        f"Be specific and detailed."
    )
    return {
        "outline": response.content,
        "stages_run": state.get("stages_run", []) + ["outline"],
    }


def check_outline_quality(state: State) -> str:
    """Gate — Does the outline have at least 3 sections?"""
    outline = state["outline"]
    section_count = len(re.findall(r"\n#+\s|\n\d+\.", outline))
    has_sections = section_count >= 3 or outline.count("\n-") >= 6
    result = "Pass" if has_sections else "Fail"
    print(f"  [Gate] Outline quality check → {result} ({section_count} sections found)")
    return result


def refine_outline_node(state: State) -> dict:
    """Stage 2a — Outline was too thin, enrich it."""
    print(" [2a] Refining outline (was too sparse)...")
    response = llm.invoke(
        f"This blog outline needs more depth. Expand it with 2 more sections "
        f"and add 3 bullet points per section.\n\nCurrent outline:\n{state['outline']}"
    )
    return {
        "refined_outline": response.content,
        "stages_run": state["stages_run"] + ["refine_outline"],
    }


def draft_node(state: State) -> dict:
    """Stage 2/3 — Write the full draft from the outline."""
    print("[2/3] Writing draft...")
    active_outline = state.get("refined_outline") or state["outline"]
    response = llm.invoke(
        f"Write a complete blog post based on this outline.\n"
        f"Target audience: {state['audience']}\n"
        f"Use an engaging, conversational tone. Include a strong intro and clear conclusion.\n\n"
        f"Outline:\n{active_outline}"
    )
    return {
        "draft": response.content,
        "stages_run": state["stages_run"] + ["draft"],
    }


def check_draft_length(state: State) -> str:
    """Gate — Is the draft at least 300 words?"""
    word_count = len(state["draft"].split())
    result = "Pass" if word_count >= 300 else "Fail"
    print(f"  [Gate] Draft length check → {result} ({word_count} words)")
    return result


def expand_node(state: State) -> dict:
    """Stage 3a — Draft too short, expand with examples."""
    print("[3a] Expanding draft (was too short)...")
    response = llm.invoke(
        f"This blog post draft is too short. Add a real-world example, "
        f"a practical tip section, and expand each paragraph with more detail.\n\n"
        f"Draft:\n{state['draft']}"
    )
    return {
        "expanded_draft": response.content,
        "stages_run": state["stages_run"] + ["expand"],
    }

def seo_node(state: State) -> dict:
    """Stage 4 — Add SEO metadata and finalize."""
    print("[4/5] Adding SEO polish and finalizing...")
    active_draft = state.get("expanded_draft") or state["draft"]
    response = llm.invoke(
        f"Polish this blog post for SEO and readability:\n"
        f"1. Suggest a compelling SEO title and meta description (160 chars)\n"
        f"2. Add 5 relevant hashtags\n"
        f"3. Suggest internal linking opportunities\n"
        f"Keep your response as: SEO TIPS section first, then FINAL POST.\n\n"
        f"Draft:\n{active_draft}"
    )
    # Split SEO tips from the final post
    content = response.content
    if "FINAL POST" in content.upper():
        parts = re.split(r"FINAL POST[:\s]*", content, flags=re.IGNORECASE, maxsplit=1)
        seo_tips  = parts[0].strip()
        final_post = parts[1].strip() if len(parts) > 1 else active_draft
    else:
        seo_tips   = content
        final_post = active_draft

    return {
        "seo_tips":   seo_tips,
        "final_post": final_post,
        "stages_run": state["stages_run"] + ["seo"],
    }


workflow = StateGraph(State)

workflow.add_node("outline_node",        outline_node)
workflow.add_node("refine_outline_node", refine_outline_node)
workflow.add_node("draft_node",          draft_node)
workflow.add_node("expand_node",         expand_node)
workflow.add_node("seo_node",            seo_node)

workflow.add_edge(START, "outline_node")

workflow.add_conditional_edges(
    "outline_node",
    check_outline_quality,
    {"Fail": "refine_outline_node", "Pass": "draft_node"},
)

workflow.add_edge("refine_outline_node", "draft_node")

workflow.add_conditional_edges(
    "draft_node",
    check_draft_length,
    {"Fail": "expand_node", "Pass": "seo_node"},
)

workflow.add_edge("expand_node", "seo_node")
workflow.add_edge("seo_node",    END)

pipeline = workflow.compile()

def run(topic: str, audience: str = "general tech readers") -> None:
    print(f"\n{'═'*65}")
    print(f"  TOPIC    : {topic}")
    print(f"  AUDIENCE : {audience}")
    print(f"{'─'*65}")

    result = pipeline.invoke({
        "topic":      topic,
        "audience":   audience,
        "stages_run": [],
    })

    print(f"\n{'─'*65}")
    print(f"  PIPELINE STAGES RUN: {' → '.join(result['stages_run'])}")
    print(f"{'─'*65}")

    if result.get("refined_outline"):
        print(f"\nREFINED OUTLINE:\n{result['refined_outline'][:300]}...")

    print(f"\nFINAL POST (excerpt):\n{result['final_post'][:600]}...")
    print(f"\nSEO TIPS:\n{result['seo_tips'][:400]}...")

if __name__ == "__main__":
    print("\n  SEQUENTIAL PIPELINE DEMO — Blog Post Writer\n")

    run(
        topic    = "Why every developer should learn LangGraph in 2026",
        audience = "mid-level Python developers interested in AI agents",
    )