import importlib.util
import inspect
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import BaseTool, tool
from langchain.agents import create_agent

"""
each skill lives in its own directory under skills/:

    skills/
     - [skill_name]/
       - prompt.md   ← expert system prompt
       - tools.py    ← @tool functions (optional)

When the agent calls load_skill("write_sql"):
  1. prompt.md  → injected as expert context
  2. tools.py   → imported; every @tool becomes callable immediately
"""

load_dotenv()

SKILLS_DIR = Path(__file__).parent / "skills"

# Tools registered at runtime by load_skill()
_extra_tools: dict[str, BaseTool] = {}

def _load_tools_from(skill_dir: Path) -> list[BaseTool]:
    """Import tools.py from a skill directory and return all @tool objects."""
    py_file = skill_dir / "tools.py"
    if not py_file.exists():
        return []
    module_id = f"skills.{skill_dir.name}"
    if module_id in sys.modules:
        return []                           # already loaded
    spec = importlib.util.spec_from_file_location(module_id, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_id] = mod
    spec.loader.exec_module(mod)
    return [obj for _, obj in inspect.getmembers(mod) if isinstance(obj, BaseTool)]

@tool
def list_skills() -> str:
    """List every available skill and whether it ships Python tools."""
    lines = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if d.is_dir():
            has_tools = "🔧 prompt + tools" if (d / "tools.py").exists() else "📄 prompt only"
            lines.append(f"  • {d.name}  [{has_tools}]")
    return "\n".join(lines) or "No skills found."

@tool
def load_skill(skill_name: str) -> str:
    """
    Load a skill by name.  Returns the expert prompt and registers
    any Python tools so the agent can call them straight away.

    Args:
        skill_name: Directory name under skills/ (e.g. 'write_sql').
    """
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.is_dir():
        available = [d.name for d in SKILLS_DIR.iterdir() if d.is_dir()]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"

    prompt = (skill_dir / "prompt.md").read_text(encoding="utf-8")

    new_tools = _load_tools_from(skill_dir)
    for t in new_tools:
        _extra_tools[t.name] = t

    tool_note = (
        f"\n\n🔧 Tools now available: {', '.join(t.name for t in new_tools)}"
        if new_tools else ""
    )
    return prompt + tool_note

LLM = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.2)

SYSTEM_PROMPT = (
    "You are a versatile expert assistant with specialist skills. "
    "When a task matches a skill, ALWAYS call load_skill first, then follow "
    "the returned prompt rules exactly and use any registered tools when helpful. "
    "If unsure which skill fits, call list_skills first."
)

def _agent():
    return create_agent(
        model=LLM,
        tools=[list_skills, load_skill, *_extra_tools.values()],
        system_prompt=SYSTEM_PROMPT,
    )

def ask(query: str) -> None:
    print(f"\n{'═'*60}\nUSER: {query}\n{'─'*60}")
    result = _agent().invoke({"messages": [{"role": "user", "content": query}]})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    # SQL skill
    ask(
        "Write a SQL query to find the top 5 customers by total revenue "
        "in the last 90 days, with order count and average order value. "
        "Tables: orders(id, customer_id, total, created_at), customers(id, name, email)."
    )

    # Legal skill
    ask(
        "Review this clause: 'The Vendor shall not be liable for any indirect, "
        "incidental, or consequential damages. Total liability shall not exceed "
        "fees paid in the last 30 days. Agreement auto-renews annually unless "
        "cancelled with 90 days written notice.'"
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

    ask("What can you help me with?")