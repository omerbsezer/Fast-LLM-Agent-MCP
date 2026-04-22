import re
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from dotenv import load_dotenv
load_dotenv()

"""
The key to seeing redaction/masking is to iterate through ALL messages
in the result — the middleware modifies messages at each stage (input,
tool results, output), which is only visible when you inspect each message.
"""

@tool
def customer_lookup(email: str) -> str:
    """Look up a customer account by email."""
    db = {
        "john.doe@example.com": {"name": "John Doe", "card": "4111 1111 1111 1234", "status": "active"},
        "alice@example.com":    {"name": "Alice Smith", "card": "5500 0000 0000 4444", "status": "active"},
    }
    record = db.get(email.lower())
    if not record:
        return f"No account found for {email}"
    return f"Name: {record['name']}\nEmail: {email}\nCard: {record['card']}\nStatus: {record['status']}"

llm = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0)

agent = create_agent(
    model=llm,
    tools=[customer_lookup],
    middleware=[
        PIIMiddleware("email",       strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask",   apply_to_input=True, apply_to_output=True, apply_to_tool_results=True),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block", apply_to_input=True),
    ],
)

def ask(user_input: str) -> None:
    print(f"\n{'─'*60}")
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"👤 USER  : {msg.content}")
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(c.get("text","") for c in content if isinstance(c, dict))
                content = re.sub(r"<thinking>.*?</thinking>\s*", "", str(content), flags=re.DOTALL).strip()
                if content:
                    print(f"🤖 AGENT : {content}")
            elif isinstance(msg, ToolMessage):
                print(f"🛠️ TOOL  : {msg.content}")
    except Exception as e:
        print(f"🚫 BLOCK : {e}")

if __name__ == "__main__":
    ask("Hi, my email is john.doe@example.com, look up my account.")
    ask("Look up alice@example.com and show me her card details.")
    ask("My API key is sk-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234, help me.")