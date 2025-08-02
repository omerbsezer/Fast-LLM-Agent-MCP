## Multi-Agent Sequential, Streamlit GUI

It shows how to implement multi-agent sequential workflow.

Link:  https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/06-multiagent-workflow-sequential

![sample-06](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/06-multiagent-workflow-sequential/gif/multi-agent-sequential.gif)


The agents will run in the order provided: 

```
Writer -> Reviewer -> Refactorer
```

Please add .env with Gemini  

``` 
# agent/.env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
``` 

## Install, Update Libraries
Install first time:
```
pip install -r requirements.txt
```

Update:
```
pip install --upgrade fastapi uvicorn google-adk google-generativeai
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
- I want to create a space shuttle game where meteors fall from the top of the screen. The player controls the shuttle using the arrow keys and can shoot at the meteors by pressing the spacebar.
- I want to create a number guessing game where the app randomly selects a number between 0 and 100. The player tries to guess the number, and after each guess, the app provides feedback, saying 'Go higher' if the guess is too low, and 'Go lower' if the guess is too high"
```
