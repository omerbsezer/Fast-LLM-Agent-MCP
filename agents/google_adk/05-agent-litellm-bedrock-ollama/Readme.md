## Agent LiteLLM, Bedrock, Ollama, Streamlit UI

Please add .env with AWS Bedrock or use .aws/config, .aws/credentials

For only AWS Bedrock:
``` 
# agent/.env
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION_NAME = ""
``` 

For OpenAI, Antropic:
- https://google.github.io/adk-docs/agents/models/#using-cloud-proprietary-models-via-litellm

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
- which llm model is running in the background?
```
