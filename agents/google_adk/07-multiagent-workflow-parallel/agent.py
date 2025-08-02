from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search
import uvicorn

# Load environment variables
load_dotenv()

MODEL = "gemini-2.5-flash"

APP_NAME = "search_memory_app"
USER_ID = "user123"
SESSION_ID = "session123"

# --- Agent Definitions ---
# the Pydantic model for the output of the TopicSetterAgent
class TopicOutput(BaseModel):
    subtopic_1: str
    subtopic_2: str
    subtopic_3: str

topic_setter = LlmAgent(
    name="TopicSetterAgent",
    model=MODEL,
    instruction="""
    You are a research planner. Your task is to break the user's input query into three distinct and relevant subtopics.
    Respond ONLY in this format:

    {
    "subtopic_1": "...",
    "subtopic_2": "...",
    "subtopic_3": "..."
    }
    """,
    output_schema=TopicOutput,
    output_key="topic_output"
)

researcher_agent_1 = LlmAgent(
    name="SubResearcherOne",
    model=MODEL,
    instruction="""
    You are a research assistant. Research the subtopic: "{topic_output.subtopic_1}".
    Summarize your findings in 1-2 sentences.
    """,
    tools=[google_search],
    output_key="result_1"
)

researcher_agent_2 = LlmAgent(
    name="SubResearcherTwo",
    model=MODEL,
    instruction="""
    You are a research assistant. Research the subtopic: "{topic_output.subtopic_2}".
    Summarize your findings in 1-2 sentences.
    """,
    tools=[google_search],
    output_key="result_2"
)

researcher_agent_3 = LlmAgent(
    name="SubResearcherThree",
    model=MODEL,
    instruction="""
    You are a research assistant. Research the subtopic: "{topic_output.subtopic_3}".
    Summarize your findings in 1-2 sentences.
    """,
    tools=[google_search],
    output_key="result_3"
)

# final synthesis
merger_agent = LlmAgent(
    name="ResearchSynthesizer",
    model=MODEL,
    instruction="""
    You are a synthesis assistant. Merge the three research summaries below into a single cohesive research report.

    Subtopic 1 ({topic_output.subtopic_1}): {result_1}
    Subtopic 2 ({topic_output.subtopic_2}): {result_2}
    Subtopic 3 ({topic_output.subtopic_3}): {result_3}

    Write in this format:

    ## Research Summary on the Given Topic

    ### {topic_output.subtopic_1}
    [result_1]

    ### {topic_output.subtopic_2}
    [result_2]

    ### {topic_output.subtopic_3}
    [result_3]

    ### Conclusion
    Write 2-3 sentences summarizing the topic overall.
    """
)

parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="Runs multiple research agents in parallel to gather information."
)

research_pipeline = SequentialAgent(
    name="GeneralResearchPipeline",
    description="Extracts a topic, researches it in parallel, then synthesizes a report.",
    sub_agents=[topic_setter, parallel_research_agent, merger_agent],
)

root_agent = research_pipeline

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    """
    Initializes the session and runner on application startup.
    """
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    global runner
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )

@app.post("/ask")
async def ask_agent(req: QueryRequest):
    """
    Endpoint to send a query to the agent pipeline.
    """
    content = types.Content(role="user", parts=[types.Part(text=req.query)])
    # `runner.run()` is a generator, not a coroutine, so it should not be awaited.
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    responses = []
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            responses.append(event.content.parts[0].text)

    # Await both the get_session and add_session_to_memory calls.
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    await memory_service.add_session_to_memory(session)

    return {"responses": responses or ["No response received."]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
