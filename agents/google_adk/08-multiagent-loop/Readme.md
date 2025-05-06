## Multi-Agent Loop Streamlit UI

The agents will run in the order provided: 

```
Loop runs: Refiner -> Checker -> StopChecker
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
- I want to create a game, tic toc toe
- I want to create a game, space shuttle that shoots the rocks that are coming from top to down
```
