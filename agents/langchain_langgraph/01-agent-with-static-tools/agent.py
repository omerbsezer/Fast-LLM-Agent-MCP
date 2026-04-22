from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent 
from dotenv import load_dotenv
load_dotenv()

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock)."""
    mock_data = {
        "istanbul": "Partly cloudy, 18°C",
        "berlin":   "Rainy, 10°C",
        "london":   "Overcast, 13°C",
    }
    return mock_data.get(city.lower(), f"Weather data not available for {city}.")


llm = ChatBedrockConverse(
    model="us.amazon.nova-pro-v1:0",
    temperature=0,
    # AWS credentials are read from env vars or ~/.aws/credentials:
    #   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
)

tools = [add, multiply, get_weather]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools when needed.",
)

def run(query: str) -> str:
    """Send a query to the agent and return the final answer."""
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    questions = [
        "What is 42 multiplied by 7?",
        "What is the weather in Istanbul?",
        "Add 123 and 456, then multiply the result by 2.",
    ]

    for q in questions:
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print(f"A: {run(q)}")