from praisonaiagents import Agent, MCP
import ollama
import re
import multiprocessing
from dotenv import load_dotenv
import os
import json
import multiprocessing
from multiprocessing import Queue
from typing import Callable, Any, Dict

load_dotenv()  # Load variables from .env file

LLM_MODEL="ollama/llama3.2:1b"
# LLM_MODEL="ollama/llama3.2:3b"
# LLM_MODEL="ollama/llama3.1:8b"
# LLM_MODEL="ollama/gemma3:1b"
# LLM_MODEL="ollama/gemma3:4b"
# LLM_MODEL="ollama/deepseek-r1:8b"

def initialize_airbnb_agent():
    return Agent(
        # instructions="""You are a travel planning assistant that specializes in finding Airbnb listings and retrieving detailed information about specific properties. 
        # Your goal is to help users effortlessly plan their trips by providing relevant and structured data—without requiring an API key.

        # You can:
        # - Use **airbnb_search** to find Airbnb listings based on filters like location, price range, and number of guests. Always include direct links to the listings.
        # - Use **airbnb_listing_details** to get comprehensive details about a specific property, including pricing, amenities, reviews, and availability. Always include direct links to the listings.

        # Use the tools wisely to assist users in making informed travel decisions while respecting Airbnb’s guidelines.
        # """,
        instructions="""You are a research assistant specializing in Airbnb data analysis. Your role is to process MCP responses and provide structured, actionable insights. Follow these steps:
        1. For **airbnb_search** results:
        - Use 'searchUrl' as the base for all listing links.
        - Extract pricing details from 'structuredDisplayPrice':
        * 'primaryLine.accessibilityLabel' for the nightly rate.
        * 'explanationData.priceDetails' for a full price breakdown.
        - Parse 'avgRatingA11yLabel' for review scores.
        - Combine 'checkin' and 'checkout' dates from search parameters.
        2. Structure your output to include:
        - A direct search URL preserving filters.
        - A price comparison table showing the base rate vs. the total cost.
        - Rating summaries with review counts.
        - Clear date availability indicators.
        Ensure accurate price calculations and provide actionable insights for users.""",
        llm=LLM_MODEL,
        tools=MCP("npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt", config={"response_format": "structured"})
    )

def initialize_youtube_transcript_agent():
    return Agent(
        instructions="""You are a research assistant that understands YouTube transcript JSON structures. When receiving MCP tool responses:
        1. For **get_transcripts** results:
        - Access 'transcripts' array for text segments
        - Use 'text', 'start' (seconds), and 'duration' from each segment
        - Check 'subtitles' for alternative language tracks
        - Use 'timestamps' to reference video positions
        2. For complex analyses:
        - Extract 'headings' for section markers
        - Use 'key_points' for important highlights
        3. Always structure your output as:
        - Summary of main content themes
        - Timestamped key excerpts
        - Language availability notice
        - Direct video link reference
        Focus on accurate timestamp referencing...""",
        llm=LLM_MODEL,
        tools=MCP("npx -y @sinco-lab/mcp-youtube-transcript", config={"response_format": "structured"})
    )


def initialize_google_serper_agent():
    serper_api_key = os.getenv("SERPER_API_KEY")
    return Agent(
        instructions="""You are a research assistant that understands JSON structures. When receiving MCP tool responses:
        1. For **google_search** results:
        - Access 'organic' array for web results
        - Use 'title', 'link', and 'snippet' from each item
        - Check 'knowledgeGraph' for entity information
        - Use 'faqs' for question-answer pairs
        2. For **scrape** results:
        - Use 'content' field for page text
        - Extract 'headings' and 'key_points' if available
        3. Always structure your output as:
        - Summary of key findings
        - Bullet points with relevant links
        - Highlight special sections like FAQs
        Now perform web searches when needed...""",
        llm=LLM_MODEL,
        tools=MCP("npx -y serper-search-scrape-mcp-server", env={"SERPER_API_KEY": serper_api_key}, config={"response_format": "structured"})
    )

def initialize_tavily_agent():
    print("tavily agent")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    return Agent(
        instructions="""You are a research assistant that understands Tavily's JSON structures. When receiving MCP responses:
        1. For **tavily-search** results:
        - Access 'results' array for web content
        - Use 'title', 'url', and 'content' from each item
        - Check 'answer' for direct answers when available
        2. For **tavily-extract** results:
        - Use 'content' for main text
        - Extract 'metadata' for publication details
        - Use 'links' for related resources
        3. Always structure output as:
        - Summary of key points with timeliness emphasis
        - Source citations with dates
        - Highlighted quotes or statistics
        Keep it simple.""",
        llm=LLM_MODEL,
        tools=MCP("npx -y tavily-mcp@0.1.4", env={"TAVILY_API_KEY": tavily_api_key}, config={"response_format": "structured"})
    )

# to run the agent in a separate process
def agent_worker(agent_initializer, query, queue):
    try:
        agent = agent_initializer()
        result = agent.start(query)
        queue.put(result)
    except Exception as e:
        queue.put(f"Error: {e}")

# to call an agent in a separate process
def run_agent_in_process(agent_initializer, query: str) -> str:
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=agent_worker, args=(agent_initializer, query, queue))
    p.start()
    p.join()  # wait for the process to finish.
    return queue.get()

def route_query(query: str) -> str:
    keywords = {
        "airbnb": initialize_airbnb_agent,
        "google": initialize_google_serper_agent,
        "tavily": initialize_tavily_agent,
        "youtube": initialize_youtube_transcript_agent,
        "serper": initialize_google_serper_agent
    }

    for key, agent_initializer in keywords.items():
        if re.search(rf"\b{key}\b", query, re.IGNORECASE):
            if key in ["airbnb", "google", "youtube", "tavily", "serper"]:
                enhanced_query = f"{query} [Respond using structured JSON response from MCP]"
                return run_agent_in_process(agent_initializer, enhanced_query)
            else:
                return ollama.generate(model=LLM_MODEL, prompt=query)["response"]

    return ollama.generate(model=LLM_MODEL, prompt=query)["response"]
