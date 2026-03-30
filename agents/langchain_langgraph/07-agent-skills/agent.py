import importlib.util
import inspect
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain.agents import create_agent

load_dotenv()

SKILLS_DIR = Path(__file__).parent / "skills"
LLM = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0.2)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a versatile expert assistant with specialist skills.\n\n"
    "SKILL ROUTING — call load_skill() with exactly one of these names:\n"
    "  • 'write_sql'        → user wants to WRITE or GENERATE a SQL query\n"
    "  • 'review_legal_doc' → user wants to REVIEW a CONTRACT, CLAUSE, or LEGAL text\n"
    "  • 'code_review'      → user wants to REVIEW SOURCE CODE (any language)\n"
    "  • call list_skills() → if unsure\n\n"
    "IMPORTANT: tool names like 'detect_sql_risks', 'score_legal_risk' etc. are NOT skill names.\n"
    "After load_skill() returns, use any registered tools to enrich your answer."
))

def _import_tools(skill_dir: Path) -> list[BaseTool]:
    """Import tools.py from a skill directory and return all @tool objects."""
    py_file = skill_dir / "tools.py"
    if not py_file.exists():
        return []
    module_id = f"skills.{skill_dir.name}"
    if module_id not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_id, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_id] = mod
        spec.loader.exec_module(mod)
    return [obj for _, obj in inspect.getmembers(sys.modules[module_id]) if isinstance(obj, BaseTool)]

@tool
def list_skills() -> str:
    """List every available skill."""
    lines = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if d.is_dir():
            tag = "🔧 prompt + tools" if (d / "tools.py").exists() else "📄 prompt only"
            lines.append(f"  • {d.name}  [{tag}]")
    return "\n".join(lines) or "No skills found."


def _make_load_skill(session_tools: dict[str, BaseTool]):
    """Return a load_skill tool that registers into the given session dict."""

    @tool
    def load_skill(skill_name: str) -> str:
        """ Load a specialist skill by its directory name."""
        skill_dir = SKILLS_DIR / skill_name
        if not skill_dir.is_dir():
            available = [d.name for d in SKILLS_DIR.iterdir() if d.is_dir()]
            return f"Skill '{skill_name}' not found. Valid names: {', '.join(available)}"

        prompt = (skill_dir / "prompt.md").read_text(encoding="utf-8")
        new_tools = _import_tools(skill_dir)
        session_tools.update({t.name: t for t in new_tools})

        tool_note = (
            "\n\n🔧 Tools registered (call directly, NOT via load_skill):\n"
            + "\n".join(f"  - {t.name}" for t in new_tools)
            if new_tools else ""
        )
        return prompt + tool_note

    return load_skill

def ask(query: str) -> None:
    print(f"\n{'═'*60}\nUSER: {query}\n{'─'*60}")

    # fresh tool scope per call
    session_tools: dict[str, BaseTool] = {}
    load_skill = _make_load_skill(session_tools)

    def build_agent():
        return create_agent(
            model=LLM,
            tools=[list_skills, load_skill, *session_tools.values()],
            system_prompt=SYSTEM_PROMPT,
        )

    result = build_agent().invoke({"messages": [{"role": "user", "content": query}]})

    for msg in result["messages"]:
        for tc in getattr(msg, "tool_calls", []):
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
            print(f"  🔧 {tc['name']}({args_str})")
        if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", []):
            print(f"\n{msg.content}")
    print()

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