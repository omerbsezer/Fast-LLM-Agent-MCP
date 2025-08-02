from fastapi import FastAPI, Request, Response, Cookie
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator, current_time, python_repl
import sqlite3
import json
import uuid
import os

MODEL = "us.amazon.nova-pro-v1:0"

bedrock_model = BedrockModel(
    model_id=MODEL,
    temperature=0.3,
    top_p=0.8,
)

DB_PATH = "sessions.db"

# Ensure sessions table exists
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                messages TEXT
            )
        ''')
init_db()

# Save session messages to DB
def save_session_messages(session_id: str, messages: list):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "REPLACE INTO sessions (session_id, messages) VALUES (?, ?)",
            (session_id, json.dumps(messages))
        )

# Load session messages from DB
def load_session_messages(session_id: str) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT messages FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return []

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    session_id = str(uuid.uuid4())  # always a new session

    messages = []

    agent = Agent(
        model=bedrock_model,
        system_prompt='Help user interact using available tools. '
                      '- For all other questions, respond using your own knowledge.',
        messages=messages
    )

    try:
        result = agent(query)
        save_session_messages(session_id, agent.messages)  # still saved to DB, just not reused
        return {"response": result.message}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
