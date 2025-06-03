from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator


load_dotenv()

MODEL = "gemini-2.5-flash-preview-04-17" 
# MODEL="gemini-2.5-pro-preview-03-25"
# MODEL="gemini-2.0-flash"
# MODEL="gemini-2.0-flash-lite"

APP_NAME = "search_memory_app"
USER_ID = "user123"
SESSION_ID = "session123"

# Agent to generate/refine code based on state['current_code'] and state['requirements']
code_refiner = LlmAgent(
    name="CodeRefiner",
    model=MODEL,
    instruction="Read state['current_code'] (if exists) and state['requirements']. Generate/refine Python code to meet requirements. Save to state['current_code'].",
    output_key="current_code" # Overwrites previous code in state
)

# Agent to check if the code meets quality standards
quality_checker = LlmAgent(
    name="QualityChecker",
    model=MODEL,
    instruction="Evaluate the code in state['current_code'] against state['requirements']. Output 'pass' or 'fail'.",
    output_key="quality_status"
)

# Custom agent to check the status and escalate if 'pass'
class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        status = ctx.session.state.get("quality_status", "fail")
        should_stop = (status == "pass")
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))

refinement_loop = LoopAgent(
    name="CodeRefinementLoop",
    max_iterations=5,
    sub_agents=[code_refiner, quality_checker, CheckStatusAndEscalate(name="StopChecker")]
)

root_agent = refinement_loop

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_agent(req: QueryRequest):
    content = types.Content(role="user", parts=[types.Part(text=req.query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    responses = []
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            responses.append(event.content.parts[0].text)

    # Save session to memory after each turn
    session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    memory_service.add_session_to_memory(session)

    return {"responses": responses or ["No response received."]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)