
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    mock = {"istanbul": "Sunny, 28°C", "berlin": "Cloudy, 12°C"}
    return mock.get(city.lower(), f"No data for {city}.")

llm = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0).bind_tools([get_weather])

# SystemMessage — sets the AI persona (user never sees this)
system = SystemMessage("You are a helpful weather assistant. Use tools when asked about weather.")
print(f"[SystemMessage] {system.content}\n")

# HumanMessage — the user's input 
human = HumanMessage("What's the weather in Istanbul?")
print(f"[HumanMessage] {human.content}\n")

# AIMessage — model responds, requests a tool call
ai_msg: AIMessage = llm.invoke([system, human])
print(f"[AIMessage] content='{ai_msg.content}' | tool_calls={[tc['name'] for tc in ai_msg.tool_calls]}\n")

# ToolMessage — tool result, linked to AIMessage via tool_call_id 
tool_call = ai_msg.tool_calls[0]
result = get_weather.invoke(tool_call["args"])
tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
print(f"[ToolMessage] {tool_msg.content} (tool_call_id={tool_msg.tool_call_id})\n")

# Final — model reads the ToolMessage and answers the user
final: AIMessage = llm.invoke([system, human, ai_msg, tool_msg])
print(f"[AIMessage] Final answer: {final.content}")