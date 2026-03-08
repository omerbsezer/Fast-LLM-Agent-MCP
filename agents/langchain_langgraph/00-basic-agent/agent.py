from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatBedrockConverse(
    model="us.amazon.nova-pro-v1:0",
    temperature=0,
    # AWS credentials are read from env vars or ~/.aws/credentials:
    #   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
)

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant. Use the available tools when needed."
)

def run(query: str) -> str:
    """Send a query to the agent and return the final answer."""
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    questions = [
        "Which model runs on your backend?",
    ]

    for q in questions:
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print(f"A: {run(q)}")