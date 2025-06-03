## Agent Local MCP Tool, Container, Streamlit UI

Please add .env with Gemini API Keys

``` 
# agent/.env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
``` 

## RUN AGENT

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

## PROMPTS

```
- list the files in the '/home/omer/mcp-test'
- create test2.py in the '/home/omer/mcp-test' and write 'print("Added by MCP Tool: @modelcontextprotocol/server-filesystem")' in it.
- read test2.py in the '/home/omer/mcp-test'
- run the test2.py '/home/omer/mcp-test' with "python test2.py"
- delete the test2.py '/home/omer/mcp-test'
```
