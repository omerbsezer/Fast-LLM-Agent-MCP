from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator, current_time, python_repl

MODEL = "us.amazon.nova-pro-v1:0"

bedrock_model = BedrockModel(
    model_id=MODEL,
    temperature=0.7, 
    top_p=0.9,
)

researcher_agent = Agent(
    model=bedrock_model,
    system_prompt=(
        "You are a Destination Researcher. Your primary goal is to gather detailed, high-level information for the city or "
        "location mentioned by the user. Do NOT create a day-to-day itinerary. "
        "Focus on these categories: "
        "1. **Most Attractive Places/Landmarks** (Top 5 must-sees). "
        "2. **Key Historical Facts** (3-4 fascinating historical points). "
        "3. **Best Areas to Stay** (3 distinct neighborhood suggestions with descriptions). "
        "4. **Best Local Foods** and 3 specific restaurant/street food suggestions. "
        "5. **3-5 Suggested Web Pages** for further reading (e.g., official tourism, reputable travel blogs)."
        "Report all this raw information clearly and in separate, labeled sections."
    ),
    tools=[calculator, current_time]
)

travel_guide_agent = Agent( 
    model=bedrock_model,
    system_prompt=(
        "You are a Senior Travel Guide Generator. You will receive raw destination data from a Researcher. "
        "Your goal is to synthesize this data into a structured, engaging, and comprehensive travel guide."
        "Organize the information exactly into the following sections: 'Must-See Attractions', 'Historical Highlights', 'Where to Stay', and 'Culinary Delights'. "
        "Do NOT create a 5-day itinerary. Simply structure the gathered facts into a beautiful guide format. "
        "Use Python/Calculator only if complex data aggregation is needed (unlikely for this task, but keep available)."
    ),
    tools=[python_repl, calculator]
)

writer_agent = Agent(
    model=bedrock_model,
    system_prompt=(
        "You are the Client Relations Manager. You will receive a structured travel guide. "
        "Your goal is to write a professional and clear final response to the client. "
        "Include the full structured guide and prominently feature the 'Suggested Web Pages' section at the end. "
        "Frame the guide as a 'Get Started' resource for their trip research. "
        "Keep the tone encouraging and client-focused."
    ),
    tools=[]
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: QueryRequest):
    """
    Executes a sequential travel guide workflow: Destination Researcher -> Travel Guide Generator -> Client Communicator
    """
    query = request.query
    try:
        # Step 1: Researcher (Gathers raw facts)
        research_output = researcher_agent(query, stream=False)
        
        # Step 2: Travel Guide Generator (Structures the facts)
        planner_prompt = (
            f"Generate a comprehensive travel guide based on the user's request: '{query}' "
            f"and the following raw information (DO NOT create an itinerary): {research_output}"
        )
        guide_output = travel_guide_agent(planner_prompt, stream=False)

        # Step 3: Writer (Formats the final response)
        writer_prompt = (
            f"User Inquiry: {query}\n\n"
            f"Synthesized Travel Guide Content: {guide_output}\n\n"
            "Please write the final client response (complete guide, not like email):"
        )
        final_response = writer_agent(writer_prompt, stream=False)

        return {
            "workflow_steps": {
                "raw_research_data": str(research_output),
                "structured_guide_content": str(guide_output)
            },
            "response": str(final_response)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow processing failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=8000)