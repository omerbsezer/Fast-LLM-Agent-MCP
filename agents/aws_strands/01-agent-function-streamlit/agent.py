from fastapi import FastAPI, Request
from pydantic import BaseModel
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator, current_time, python_repl

MODEL = "us.amazon.nova-pro-v1:0"

bedrock_model = BedrockModel(
    model_id=MODEL,
    temperature=0.3,
    top_p=0.8,
)

@tool
def letter_counter(word: str, letter: str) -> int:
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0
    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")
    return word.lower().count(letter.lower())

agent = Agent(
    model=bedrock_model, 
    system_prompt='Help user interact with the local filesystem using available tools. \n'
        '- For all other questions, respond using your own knowledge.',
    tools=[calculator, current_time, python_repl, letter_counter])


app = FastAPI()
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
