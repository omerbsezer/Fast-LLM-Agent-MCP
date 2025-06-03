from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.models.lite_llm import LiteLlm
import boto3

load_dotenv()

APP_NAME = "search_memory_app"
USER_ID = "user123"
SESSION_ID = "session123"

## AWS Sample with Bedrock, one-by-one enable
def create_agent() -> Agent:
    instruction = (
        "You are a helpful assistant."
    )
    return Agent(
        model=LiteLlm(model="bedrock/meta.llama3-1-405b-instruct-v1:0"),
        name="bedrock_agent",
        instruction=instruction
    )

## Ollma Sample, one-by-one enable
# def create_agent() -> Agent:
#     instruction = (
#         "You are a helpful assistant."
#     )
#     return Agent(
#         model=LiteLlm(model="ollama/llama3.2:1b"),
#         name="ollama_agent",
#         instruction=instruction
#     )

session_service = InMemorySessionService()

session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

agent = create_agent()
runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service
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

    return {"response": final_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)