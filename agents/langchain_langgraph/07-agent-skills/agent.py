from pathlib import Path
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

"""
The agent dynamically loads specialist "skills" from Markdown files.
Each skill is a focused expert prompt that sharpens the agent's behaviour
for a specific task — without needing separate agents or fine-tuning.

Available skills (in skills/ directory):
  write_sql        — Expert SQL query writer (PostgreSQL)
  review_legal_doc — Commercial contract risk reviewer
  code_review      — Principal engineer code reviewer

How it works:
  1. User sends a task
  2. Agent calls load_skill(skill_name) to fetch the expert prompt
  3. Agent re-invokes itself with the skill prompt as additional context
  4. User gets a specialist-quality response
"""

SKILLS_DIR = Path(__file__).parent / "skills"

@tool
def load_skill(skill_name: str) -> str:
    """
    Load a specialist skill prompt by name.

    Available skills:
      - write_sql        : Expert PostgreSQL query writer
      - review_legal_doc : Commercial contract risk reviewer
      - code_review      : Principal engineer code reviewer

    Returns the full skill prompt to guide your response.
    """
    path = SKILLS_DIR / f"{skill_name}.md"
    if not path.exists():
        available = [p.stem for p in SKILLS_DIR.glob("*.md")]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


@tool
def list_skills() -> str:
    """List all available skills with a short description."""
    skills = {
        "write_sql":        "Expert PostgreSQL query writer with CTEs, window functions, performance tips.",
        "review_legal_doc": "Commercial contract reviewer — flags risk clauses, missing protections.",
        "code_review":      "Principal engineer reviewer — correctness, security, performance, style.",
    }
    return "\n".join(f"  • {name}: {desc}" for name, desc in skills.items())

llm = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.2)

agent = create_agent(
    model=llm,
    tools=[list_skills, load_skill],
    system_prompt=(
        "You are a versatile expert assistant with access to specialist skills. "
        "When a user asks a task that matches a skill (SQL, legal review, code review), "
        "ALWAYS call load_skill first to load the appropriate expert prompt, "
        "then apply that skill's rules and output format precisely in your response. "
        "If unsure which skill to use, call list_skills first."
    ),
)

def ask(query: str) -> None:
    print(f"\n{'═'*65}")
    print(f"USER: {query}")
    print(f"{'─'*65}")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    # SQL skill
    ask(
        "Write a SQL query to find the top 5 customers by total revenue "
        "in the last 90 days, including their order count and average order value. "
        "Tables: orders(id, customer_id, total, created_at), customers(id, name, email)."
    )

    # Legal skill
    ask(
        "Review this contract clause: "
        "'The Vendor shall not be liable for any indirect, incidental, or consequential damages "
        "arising from the use of the Service, even if advised of the possibility of such damages. "
        "Total liability shall not exceed the fees paid in the last 30 days. "
        "This agreement auto-renews annually unless cancelled with 90 days written notice.'"
    )

    # Code review skill
    ask(
        "Review this Python function:\n\n"
        "def get_user(user_id):\n"
        "    conn = psycopg2.connect('postgresql://admin:password123@db:5432/prod')\n"
        "    cur = conn.cursor()\n"
        "    cur.execute(f\"SELECT * FROM users WHERE id = {user_id}\")\n"
        "    return cur.fetchone()"
    )

    # No clear skill — agent should call list_skills first
    ask("What can you help me with?")