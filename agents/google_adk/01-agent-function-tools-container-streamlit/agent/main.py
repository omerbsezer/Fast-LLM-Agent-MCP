import os
import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from fastapi import FastAPI
from pydantic import BaseModel

from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

MODEL = "gemini-2.5-flash"

load_dotenv()

def get_weather(city: str) -> dict:
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C (77°F)."
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

weather_tool = FunctionTool(func=get_weather)

def get_current_time(city: str) -> dict:
    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have timezone information for {city}."
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    return {"status": "success", "report": report}


time_tool = FunctionTool(func=get_current_time)

def create_agent() -> Agent:
    instruction = (
        "You are a helpful assistant that can answer general questions and provide weather and time information for cities.\n"
        "- Use 'get_weather' if the user asks about the weather.\n"
        "- Use 'get_current_time' if the user asks about the time.\n"
        "- For all other questions, respond using your own knowledge."
    )
    return Agent(
        model=MODEL,
        name="weather_time_agent",
        instruction=instruction,
        tools=[weather_tool, time_tool],
    )

# runner to create session, memory, to call agent
def init_runner(agent: Agent) -> Runner:
    APP_NAME = "weather_time_app"
    USER_ID = "user123"
    SESSION_ID = "session123"

    session_service = InMemorySessionService()
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    return Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

def call_agent(runner: Runner, user_id: str, session_id: str, query: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=query)])
    events = runner.run(user_id=user_id, session_id=session_id, new_message=content)
    for event in events:
        if event.is_final_response():
            return event.content.parts[0].text
    return "No response received."

app = FastAPI()
agent = create_agent()
runner = init_runner(agent)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_agent(req: QueryRequest):
    response = call_agent(runner, user_id="user123", session_id="session123", query=req.query)
    return {"response": response}