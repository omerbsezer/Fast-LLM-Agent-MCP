import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
import os

load_dotenv()

MODEL = "us.amazon.nova-pro-v1:0"

bedrock_model = BedrockModel(
    model_id=MODEL,
    temperature=0.3,
    top_p=0.8,
)

agent = None
mcp_client = None
serper_api_key = os.getenv("SERPER_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, mcp_client
    
    async with AsyncExitStack() as stack:
        
        mcp_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command='npx',
                args=["-y", "serper-search-scrape-mcp-server"],
                env={"SERPER_API_KEY": serper_api_key}
            )
        ))

        stack.enter_context(mcp_client)

        tools = mcp_client.list_tools_sync()
        agent = Agent(
            model=bedrock_model,
            system_prompt=(
                """You are a research assistant and search on SERPER, search and give results in link, title, content
                'You MUST:
                - Use **google_search** to perform web searches when the user is asking for general information, how-to guides, comparisons, news, or any content that could be sourced from the internet. This tool retrieves:
                    - Perform web searches using user queries to find organic results, FAQs, related searches, and knowledge graph entries.
                    - Handle a wide range of search intents: informational, comparative, educational, technical, current events, etc.
                    - Always return useful summaries along with links to the most relevant pages.
                **Tool Parameters**
                    - `q`: Required. The search query string (e.g., "how Kubernetes works", "latest AI trends 2025"). Retrieve from the prompt.
                    - `gl`: Required. Geographic region code in ISO 3166-1 alpha-2 format (e.g., "us", "de", "gb"). Use "en".
                    - `hl`: Required. Language code in ISO 639-1 format (e.g., "en", "fr", "es"). Use "en.
                    - `location`: Required. Location for search results (e.g., 'SoHo, New York, United States', 'California, United States'). Use "United States".
                Always summarize the top results clearly and include direct URLs for reference.
                - Use **scrape** to extract content from a specific webpage when:
                    - The user provides a URL and asks for content, summaries, or metadata
                    - A relevant link was previously found via **google_search** and needs to be explored further
                - Use the tools wisely to assist users. Based on the provided results, use Function tool call returns, retrieve only content.  
                - Parse the JSON response carefully and extract **relevant fields**. Give the search results with TITLE, LINK, CONTENT or SNIPPET.
                - For all other questions, respond using your own knowledge."""
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
