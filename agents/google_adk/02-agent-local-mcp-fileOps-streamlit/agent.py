import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from fastapi import FastAPI, Request


load_dotenv()

MODEL = "gemini-2.5-flash"
MCP_TOOL_PATH =  "/home/omer/mcp-test"
# # MODEL="gemini-2.5-pro-preview-03-25"


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# MCPToolset.from_server(...) is an async likely performing I/O, so all chain async
async def get_agent_async():
    """Creates an ADK Agent equipped with tools from the MCP Server."""
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem", MCP_TOOL_PATH],
        )
    )
    print(f"Fetched {len(tools)} tools from MCP server.")
    root_agent = LlmAgent(
        model=MODEL,
        name='filesystem_assistant',
        instruction='Help user interact with the local filesystem using available tools. \n'
        '- For all other questions, respond using your own knowledge.',
        tools=tools,
    )
    return root_agent, exit_stack


async def handle_query(query: str):
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()

    session = session_service.create_session(
        state={}, app_name='mcp_filesystem_app', user_id='user_fs'
    )

    content = types.Content(role='user', parts=[types.Part(text=query)])
    root_agent, exit_stack = await get_agent_async()

    runner = Runner(
        app_name='mcp_filesystem_app',
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    result = []
    async for event in events_async:
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    result.append(part.text)
        elif hasattr(event, "tool_name") and hasattr(event, "parameters"):
            result.append(f"Tool called: {event.tool_name}({event.parameters})")
        elif hasattr(event, "response"):
            result.append(f"Tool response: {event.response}")
        else:
            result.append(f"Unrecognized event: {event}")

    await exit_stack.aclose()

    final_output = "\n".join(result).strip()
    return final_output if final_output else "No response from model or tools."

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    response = await handle_query(query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)