## Agent with Memory, Google Search Built-in Tools, Streamlit UI

It shows how to run built-in tool Google Search and Agent Memory.

Link: https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/04-agent-memory-builtin-search-tool

**With Memory:**
![sample-04-with-memory](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/04-agent-memory-builtin-search-tool/gif/agent-with-memory.gif)


**Without Memory:**
![sample-04-without-memory](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/04-agent-memory-builtin-search-tool/gif/agent-without-memory.gif)

Please add .env with Gemini  

``` 
# agent/.env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
``` 

## Run Agent

Please run non-root username. 
```
uvicorn agent:app --host 0.0.0.0 --port 8000
```


## GUI
After running container, then pls run streamlit, it asks to htttp://localhost:8000/ask

```
streamlit run app.py
or
python -m streamlit run app.py
```

## Prompts

```
- hi, what is my name?
- my name is ömer, what is the llm agent?
- what is my name?
```
