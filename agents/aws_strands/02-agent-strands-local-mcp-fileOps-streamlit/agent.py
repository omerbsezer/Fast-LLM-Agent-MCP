import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI, Request
from pydantic import BaseModel
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

MODEL = "us.amazon.nova-pro-v1:0"
MCP_TOOL_PATH = "/home/omer/mcp-test"

bedrock_model = BedrockModel(
    model_id=MODEL,
    temperature=0.3,
    top_p=0.8,
)

agent = None
mcp_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, mcp_client

    async with AsyncExitStack() as stack:
        mcp_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", MCP_TOOL_PATH],
            )
        ))

        stack.enter_context(mcp_client)

        tools = mcp_client.list_tools_sync()
        agent = Agent(
            model=bedrock_model,
            system_prompt=(
                "Help user interact with the local filesystem using available tools. "
                "Fallback to LLM knowledge if necessary."
            ),
            tools=tools
        )

        yield

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    try:
        response = agent(query, stream=False)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=8000)
