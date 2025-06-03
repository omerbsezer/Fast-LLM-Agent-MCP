## First Agent with adk web
It shows first agent implementation with Google UI ADK Web and Gemini API key.

Link: https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/00-first-agent-with-adk-web

![sample-00](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/00-first-agent-with-adk-web/gif/weather-time-agent-adk.gif)

Install adk:
- pip install google-adk

Add .env file under multi_tool_agent directory

``` 
# multi_tool_agent/.env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
``` 

Run adk web

``` 
cd ./Fast-LLM/agents/google_adk/00-first-agent-with-adk-web
adk web
``` 


## Reference
https://google.github.io/adk-docs/get-started/quickstart/#env