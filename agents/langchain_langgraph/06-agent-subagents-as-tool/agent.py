import os
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain.agents import create_agent
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
load_dotenv()

"""
A main orchestrator agent delegates to 3 specialist subagents:

  main_agent
    - market_research_agent   - analyses market size, trends, competitors
    - financial_agent         - estimates costs, revenue, break-even
    - risk_agent              - identifies risks and mitigation strategies

User asks: "Evaluate my startup idea: an AI-powered meal planner"
Main agent calls all 3 subagents, then synthesises a final report.
"""

langfuse = Langfuse(
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
    host       = os.getenv("LANGFUSE_BASE_URL"),
)
langfuse_handler = CallbackHandler()

def make_llm():
    return ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.3)


# Subagent tools 
@tool
def search_market_data(sector: str) -> str:
    """Search market size and growth data for a sector."""
    data = {
        "meal planning": "Global meal-kit market: $20B (2024), CAGR 13%. Health-conscious segment growing fastest.",
        "ai health":     "AI in healthcare: $45B by 2026. Personalisation is top driver.",
        "food tech":     "FoodTech VC funding: $8B in 2023. Subscription models dominate.",
    }
    return next((v for k, v in data.items() if k in sector.lower()), "No specific data found.")

@tool
def find_competitors(niche: str) -> str:
    """Find key competitors in a niche."""
    comps = {
        "meal planner": "Noom ($400M revenue), MyFitnessPal (150M users), Whisk (acquired by Samsung). Gap: no fully AI-personalised option.",
        "ai nutrition": "Nutrino, Suggestic — both B2B focused. Consumer gap exists.",
    }
    return next((v for k, v in comps.items() if k in niche.lower()), "Competitor data not found.")

@tool
def estimate_costs(product_type: str) -> str:
    """Estimate development and operational costs."""
    costs = {
        "app":        "MVP: $80–150K. Monthly ops (infra + support): $15K. LLM API costs: $0.02/user/day.",
        "saas":       "MVP: $100–200K. Monthly ops: $20K. Customer acquisition: $30–80 per user.",
        "subscription": "Churn benchmark: 5–8%/month. LTV target: >3x CAC.",
    }
    return next((v for k, v in costs.items() if k in product_type.lower()), "Cost estimate not available.")

@tool
def assess_risk(area: str) -> str:
    """Assess risks in a specific area."""
    risks = {
        "regulatory": "FDA oversight if medical claims made. GDPR for EU users. Avoid 'diagnosis' language.",
        "competition":"Big Tech (Google, Apple) could replicate. Moat needed: proprietary data or partnerships.",
        "retention":  "Meal planning has high churn. Gamification and social features improve D30 retention by 40%.",
        "funding":    "Seed rounds averaging $1.5M for consumer health apps. Strong traction needed before Series A.",
    }
    return next((v for k, v in risks.items() if k in area.lower()), "Risk data not found.")


# Subagents 
market_subagent = create_agent(
    model=make_llm(),
    tools=[search_market_data, find_competitors],
    system_prompt="You are a market research specialist. Analyse market opportunity and competition. Be concise and data-driven.",
)

financial_subagent = create_agent(
    model=make_llm(),
    tools=[estimate_costs],
    system_prompt="You are a startup financial analyst. Estimate costs, revenue potential, and break-even timelines. Be specific with numbers.",
)

risk_subagent = create_agent(
    model=make_llm(),
    tools=[assess_risk],
    system_prompt="You are a startup risk advisor. Identify the top risks and suggest concrete mitigation strategies.",
)

# Subagents as tools for the main agent 
@tool("market_research", description="Research market size, trends, and competitors for a startup idea.")
def call_market_agent(query: str) -> str:
    result = market_subagent.invoke({
            "messages": [{"role": "user", "content": query}]
        }
    )
    return result["messages"][-1].content

@tool("financial_analysis", description="Estimate costs, revenue, and financial viability of a startup idea.")
def call_financial_agent(query: str) -> str:
    result = financial_subagent.invoke({
            "messages": [{"role": "user", "content": query}]
        }
    )
    return result["messages"][-1].content

@tool("risk_assessment", description="Identify key risks and mitigation strategies for a startup idea.")
def call_risk_agent(query: str) -> str:
    result = risk_subagent.invoke({
            "messages": [{"role": "user", "content": query}]
        }
    )
    return result["messages"][-1].content

main_agent = create_agent(
    model=make_llm(),
    tools=[call_market_agent, call_financial_agent, call_risk_agent],
    system_prompt=(
        "You are a senior startup advisor. When asked to evaluate a startup idea, "
        "delegate to all three specialist subagents (market research, financial analysis, risk assessment), "
        "then synthesise their findings into a clear, structured investment brief with a final verdict."
    ),
)

def evaluate(idea: str) -> None:
    print(f"\n{'═'*65}")
    print(f"IDEA: {idea}")
    print(f"{'═'*65}")
    result = main_agent.invoke({
            "messages": [{"role": "user", "content": f"Evaluate this startup idea: {idea}"}]
        },
        {
            "callbacks": [langfuse_handler]   # if you don't want to use langfuse, remove callbacks and handler
        }
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    evaluate("An AI-powered personalised meal planner app with weekly grocery delivery integration.")