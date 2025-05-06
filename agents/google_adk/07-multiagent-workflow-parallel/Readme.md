## Multi-Agent Parallel and Merger,  Streamlit UI

The agents will run in the order provided: 

```
                                 Research Agent1 
Prompt -> Topic Setter Agent ->  Research Agent2  -> Merger Agent -> Result
                                 Research Agent3 
```

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
- I want to research about the "LLM Agents"
```
