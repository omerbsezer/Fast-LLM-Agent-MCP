
from typing import TypedDict, Literal
from pydantic import BaseModel
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv
load_dotenv()

"""
A supervisor router classifies incoming queries and dispatches them
to the right specialist agent using LangGraph's Command(goto=...).

Specialist agents:
  billing_agent    - invoices, payments, subscriptions
  technical_agent  - bugs, errors, crashes, performance
  hr_agent         - leave, payroll, onboarding, policies
  general_agent    - fallback for anything else

Flow:
  User query
    - router_node  (LLM classifies → Command(goto=agent))
          - billing_agent
          - technical_agent
          - hr_agent
          - general_agent
                - END
"""

class State(TypedDict):
    query: str
    agent: str       # which agent was selected
    response: str    # final answer

# Structured classification output
AGENTS = Literal["billing_agent", "technical_agent", "hr_agent", "general_agent"]

class Classification(BaseModel):
    """Router decision."""
    agent: AGENTS
    reason: str
    confidence: float   # 0.0 – 1.0


# Specialist tools
@tool
def get_invoice(invoice_id: str) -> str:
    """Fetch invoice details by ID."""
    db = {
        "INV-001": "Invoice INV-001: $1,200 — due 2026-03-15 — UNPAID",
        "INV-002": "Invoice INV-002: $450  — due 2026-02-28 — PAID",
    }
    return db.get(invoice_id.upper(), f"Invoice {invoice_id} not found.")

@tool
def get_subscription(email: str) -> str:
    """Look up a subscription by user email."""
    return f"Subscription for {email}: Pro Plan, renews 2026-06-01, auto-renewal ON."

@tool
def lookup_error_code(code: str) -> str:
    """Look up a known error code in the knowledge base."""
    errors = {
        "ERR_500": "Internal server error. Check application logs and restart the service.",
        "ERR_403": "Forbidden. Verify user permissions and API key scopes.",
        "DB_CONN":  "Database connection failed. Check DB_HOST env var and firewall rules.",
    }
    return errors.get(code.upper(), f"Error code '{code}' not in knowledge base — escalate to L2.")

@tool
def get_leave_balance(employee_id: str) -> str:
    """Get remaining leave balance for an employee."""
    return f"Employee {employee_id}: 12 days annual leave, 3 days sick leave remaining."

@tool
def get_policy(topic: str) -> str:
    """Retrieve an HR policy document."""
    policies = {
        "remote":    "Remote work policy: up to 3 days/week, manager approval required.",
        "expenses":  "Expense policy: submit within 30 days, receipts required over $50.",
        "onboarding":"Onboarding: IT setup day 1, buddy assigned day 2, 90-day review.",
    }
    return next((v for k, v in policies.items() if k in topic.lower()), f"No policy found for '{topic}'.")


# Specialist agents
def make_llm(): return ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0)
billing_agent  = create_agent(make_llm(), [get_invoice, get_subscription],
                               system_prompt="You are a billing support specialist. Help with invoices, payments, and subscriptions.")
technical_agent = create_agent(make_llm(), [lookup_error_code],
                               system_prompt="You are a technical support engineer. Diagnose errors and provide step-by-step fixes.")
hr_agent       = create_agent(make_llm(), [get_leave_balance, get_policy],
                               system_prompt="You are an HR helpdesk assistant. Answer leave, payroll, and policy questions.")
general_agent  = create_agent(make_llm(), [],
                               system_prompt="You are a friendly general support agent. Answer questions or escalate if needed.")

AGENT_MAP = {
    "billing_agent":   billing_agent,
    "technical_agent": technical_agent,
    "hr_agent":        hr_agent,
    "general_agent":   general_agent,
}


router_llm = make_llm().with_structured_output(Classification)

def router_node(state: State) -> Command[AGENTS]:
    """Classify the query and route to the correct specialist agent."""
    classification: Classification = router_llm.invoke(
        f"""Classify this helpdesk query and route it to the right agent.

Agents:
  billing_agent   — invoices, payments, subscriptions, refunds, pricing
  technical_agent — bugs, errors, crashes, performance, outages, setup
  hr_agent        — leave, payroll, onboarding, policies, benefits
  general_agent   — anything else, greetings, unclear queries

Query: {state['query']}"""
    )

    print(f"\n  🔀 ROUTER → {classification.agent}")
    print(f"     reason     : {classification.reason}")
    print(f"     confidence : {classification.confidence:.0%}")

    return Command(
        goto=classification.agent,
        update={"agent": classification.agent},
    )


def make_agent_node(name: str):
    def node(state: State) -> Command[Literal["__end__"]]:
        agent = AGENT_MAP[name]
        result = agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
        return Command(goto=END, update={"response": result["messages"][-1].content})
    node.__name__ = name
    return node

graph = (
    StateGraph(State)
    .add_node("router",          router_node)
    .add_node("billing_agent",   make_agent_node("billing_agent"))
    .add_node("technical_agent", make_agent_node("technical_agent"))
    .add_node("hr_agent",        make_agent_node("hr_agent"))
    .add_node("general_agent",   make_agent_node("general_agent"))
    .add_edge(START, "router")
    .compile()
)

def ask(query: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  USER: {query}")
    result = graph.invoke({"query": query})
    print(f"\n  [{result['agent'].upper()}]\n  {result['response']}")

if __name__ == "__main__":
    print("\n  ROUTER AGENT DEMO — IT Helpdesk\n")

    ask("I can't find my invoice INV-001, can you help?")
    ask("I'm getting ERR_500 on the dashboard, what should I do?")
    ask("How many days of annual leave do I have left? My ID is EMP-042.")
    ask("What's the remote work policy?")
    ask("Hi, I'm new here — what do I do first?")