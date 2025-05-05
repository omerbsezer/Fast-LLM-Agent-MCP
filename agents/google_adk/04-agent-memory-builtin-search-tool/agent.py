from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, google_search
load_dotenv()

# --- App/Session Config ---
APP_NAME = "search_memory_app"
USER_ID = "user123"
SESSION_ID = "session123"

MODEL = "gemini-2.5-flash-preview-04-17"
#MODEL="gemini-2.0-flash"

def create_agent() -> Agent:
    instruction = (
        "You are a helpful assistant. Search with 'google_search'."
    )
    return Agent(
        model=MODEL,
        name="google_search_agent",
        instruction=instruction,
        tools=[google_search],
    )

# --- Init Services ---
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Create session once
session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

agent = create_agent()
runner = Runner(
    agent=agent,
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

    final_response = "No response received."
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            break

    # Save session to memory after each turn
    session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    memory_service.add_session_to_memory(session)

    return {"response": final_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)