## Agent Local MCP Tool, Container, Streamlit UI

It shows how to run local MCP tool with Gemini.

Link: https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/02-agent-local-mcp-fileOps-streamlit

![sample-02](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/02-agent-local-mcp-fileOps-streamlit/gif/agent-local-mcp-streamlit.gif)

Please add .env with Gemini API Keys

``` 
# agent/.env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
``` 

Please install nodejs, npm, npx in your system to run npx-based MCP tool.

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
- list the files in the '/home/omer/mcp-test'
- create test2.py in the '/home/omer/mcp-test' and write 'print("Added by MCP Tool: @modelcontextprotocol/server-filesystem")' in it.
- read test2.py in the '/home/omer/mcp-test'
- run the test2.py '/home/omer/mcp-test' with "python test2.py"
- delete the test2.py '/home/omer/mcp-test'
```
