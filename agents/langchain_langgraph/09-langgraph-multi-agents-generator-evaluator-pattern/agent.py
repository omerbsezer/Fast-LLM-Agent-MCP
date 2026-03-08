from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

"""
A generator LLM writes a cover letter.
An evaluator LLM scores it on 3 criteria.
If it fails any criterion, it loops back with specific feedback.
Max 3 iterations to prevent infinite loops.

Flow:
  START
    - generator  (write / rewrite with feedback)
          - evaluator  (score: tone, relevance, length)
                - [route]
                      - Accepted  → END
                      - Rejected  → generator (with feedback)
"""

generator_llm = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.8)
evaluator_llm = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0)

class State(TypedDict):
    job_title:    str
    company:      str
    candidate:    str            # brief candidate background
    cover_letter: str
    feedback:     Optional[str]
    verdict:      str            # "accepted" | "rejected"
    iteration:    int
    max_iter:     int

class Evaluation(BaseModel):
    verdict: Literal["accepted", "rejected"] = Field(
        description="Accept the cover letter only if ALL criteria pass."
    )
    tone_score: int = Field(description="Professional and engaging tone. Score 1–5.", ge=1, le=5)
    relevance_score: int = Field(description="Tailored to the specific job and company. Score 1–5.", ge=1, le=5)
    length_score: int = Field(description="Concise but complete (150–300 words). Score 1–5.", ge=1, le=5)
    feedback: Optional[str] = Field(
        description="If rejected, give specific, actionable feedback to improve the letter. Be direct."
    )


evaluator = evaluator_llm.with_structured_output(Evaluation)


def generator_node(state: State) -> dict:
    iteration = state.get("iteration", 0) + 1
    print(f"\n  [Generator] Iteration {iteration}/{state['max_iter']}")

    if state.get("feedback"):
        print(f"  Feedback applied: {state['feedback'][:80]}...")
        prompt = (
            f"Rewrite this cover letter for the {state['job_title']} role at {state['company']}.\n"
            f"Candidate background: {state['candidate']}\n\n"
            f"Previous version:\n{state['cover_letter']}\n\n"
            f"Evaluator feedback to address:\n{state['feedback']}\n\n"
            f"Write only the cover letter text, 150-300 words."
        )
    else:
        prompt = (
            f"Write a professional cover letter for a {state['job_title']} position at {state['company']}.\n"
            f"Candidate background: {state['candidate']}\n\n"
            f"Write only the cover letter text, 150-300 words."
        )

    response = generator_llm.invoke(prompt)
    return {"cover_letter": response.content, "iteration": iteration}


def evaluator_node(state: State) -> dict:
    print(f"  [Evaluator] Scoring cover letter...")

    evaluation: Evaluation = evaluator.invoke(
        f"Evaluate this cover letter for a {state['job_title']} role at {state['company']}.\n\n"
        f"Cover Letter:\n{state['cover_letter']}\n\n"
        f"Score on tone (professional), relevance (tailored to job), and length (150-300 words). "
        f"Accept only if ALL scores are 4 or above."
    )

    print(f"  Scores — Tone: {evaluation.tone_score}/5 | "
          f"Relevance: {evaluation.relevance_score}/5 | "
          f"Length: {evaluation.length_score}/5 → {evaluation.verdict.upper()}")

    if evaluation.feedback:
        print(f"  Feedback: {evaluation.feedback[:100]}...")

    return {
        "verdict":  evaluation.verdict,
        "feedback": evaluation.feedback or "",
    }


def route(state: State) -> str:
    if state["verdict"] == "accepted":
        return "Accepted"
    if state["iteration"] >= state["max_iter"]:
        print(f"\n  Max iterations ({state['max_iter']}) reached — using best version so far.")
        return "Accepted"   # ship best effort after max retries
    return "Rejected"

workflow = StateGraph(State)

workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)

workflow.add_edge(START,       "generator")
workflow.add_edge("generator", "evaluator")

workflow.add_conditional_edges(
    "evaluator",
    route,
    {
        "Accepted": END,
        "Rejected": "generator",   # loop back with feedback
    },
)

pipeline = workflow.compile()

def run(job_title: str, company: str, candidate: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  JOB      : {job_title} @ {company}")
    print(f"  CANDIDATE: {candidate}")
    print(f"{'─'*65}")

    result = pipeline.invoke({
        "job_title":    job_title,
        "company":      company,
        "candidate":    candidate,
        "cover_letter": "",
        "feedback":     None,
        "verdict":      "",
        "iteration":    0,
        "max_iter":     3,
    })

    print(f"\n{'─'*65}")
    print(f"  ACCEPTED after {result['iteration']} iteration(s)\n")
    print(result["cover_letter"])

if __name__ == "__main__":
    print("\n  GENERATOR–EVALUATOR DEMO — Cover Letter Writer\n")

    run(
        job_title = "Senior Data Engineer",
        company   = "AI Company",
        candidate = (
            "5 years experience in Python and SQL, built ETL pipelines at scale, "
            "familiar with LLMs and vector databases, strong communicator."
        ),
    )