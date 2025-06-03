## Agent LiteLLM, Bedrock, Ollama, Streamlit UI

It shows how to connect Llama 3.1 405B on AWS Bedrock and Ollama Llama3.2:1b running on local PC using Litellm.

Link: https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/05-agent-litellm-bedrock-ollama

![sample-05](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/05-agent-litellm-bedrock-ollama/gif/agent-bedrock-llama3.1-405.gif)

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

For Ollama:
- Download and run model on local PC. Please take account of your GPU VRAM size for different model. If you only want to test, please use the smallest one: e.g. llama3.2:1b
- https://ollama.com/download
- Search models, views sizes, sample: https://ollama.com/library/llama3.2


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
