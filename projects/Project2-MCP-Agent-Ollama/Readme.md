# MCP Ollama Agent
- App runs GUI using Streamlit.
- To connect each MCP tools, agents are implemented for each tool. 
- PraisonAI library is used to run agent, MCP and Ollama together.
- According to the prompt, which agents will be used selected with key search. It might be better and smarter in the future.  
  - If the prompt includes "Google", "Serper", Google Serper agent will be called
  - If the prompt includes "Youtube", Youtube agent will be called 
  - If the prompt includes "Tavily", Tavily agent will be called
- To use Tavily and Serper, you must create .env file and add your APIs from Tavily and Serper.
  - https://tavily.com/ (each month 2000 request free, no credit card needed)
  - https://serper.dev/ (first 2500 request free, no credit card needed)
- It creates a generic helper that abstracts the process creation and call for any MCP-based agent. Then, simply map each keyword to its agent initializer and use that helper.


## Ollama
- To run LLM models on your local PC, you need to use Ollama.
  - Install: https://ollama.com/download
- It depends on your PC CPU/GPU Power, you can select to run which LLM model.
- If you have only CPU, you should use smaller models (e.g. Llama3.2:1b, Llama3.2:3b)
- If you have GPU, you should know the your GPU VRAM size. You can see the LLM model size on the Ollama page. 
  - Llama3.2:1b => 1.3GB
  - Llama3.2:3b => 2GB
  - Llama3.1:8b => 4.9GB
- While running the model on GPU, model size covers ~1.5x size on the VRAM. Llama3.1:8b is normally 4.9GB, but while running, it takes 6.9GB.    

## Run
- Install node, npm, npx on your PC to run MCP server app on your local pc. 
- pip install -r requirements.txt
- To use Tavily and Serper, you must create .env file and add your APIs from Tavily and Serper.
  - SERPER_API_KEY= xxxx
  - TAVILY_API_KEY= xxxx
- python -m streamlit run app.py

## MCP & Examples
- MCP Server is an adapter layer between the remote API and agent. If any change on the remote API, MCP server cannot give correct information.
- You can check whether MCP works correctly or not from Smithery (https://smithery.ai/)
  - For example, testing Airbnb MCP server using smithery: https://smithery.ai/server/@openbnb-org/mcp-server-airbnb/tools 

### Airbnb Sample Prompt
I want to book apartment using Airbnb. I will stay in Paris between 20.05.2025-30.05.2025, searching nightly min 40€ max 60€.

### Serper (Google) Sample Prompt
I want to search "how to learn LLM" on Google using Serper

### Tavily Sample Prompt
I want to search "what is MCP?" using Tavily

### Youtube Sample Prompt
Transcribe this video => https://www.youtube.com/watch?v=eur8dUO9mvE

## Demo
### Serper (Google) Demo
- **Prompt:** I want to search "how to learn LLM" on Google using Serper
  - **PrintScreen**:
    ![serper-ask-ps](https://github.com/omerbsezer/MCP-Agent-Ollama/blob/main/gif/serper-ask.png)  

  - **GIF**: 
    ![serper-ask-gif](https://github.com/omerbsezer/MCP-Agent-Ollama/blob/main/gif/serper-ask.gif)  

### Tavily Demo
- **Prompt:** I want to search "what is MCP?" using Tavily
  - **PrintScreen**:
    ![tavily-ask-ps](https://github.com/omerbsezer/MCP-Agent-Ollama/blob/main/gif/tavily-ask.png)  

  - **GIF**: 
    ![tavily-ask-gif](https://github.com/omerbsezer/MCP-Agent-Ollama/blob/main/gif/tavily-ask.gif)  

### Youtube Demo (Error)
- **Prompt:** Transcribe this video => https://www.youtube.com/watch?v=eur8dUO9mvE
  - **GIF**: 
    ![youtube-mcp-gif](https://github.com/omerbsezer/MCP-Agent-Ollama/blob/main/gif/youtube-mcp-error.gif)  

## Analysis & Future Work
- When a prompt is sent, the LLM model determines which MCP tools to use (e.g., tavily_search, tavily_extract) and calls them accordingly. The agent’s instruction or system prompt plays a crucial role in guiding this decision.
- After receiving a JSON response from the MCP server, the LLM and agent sometimes fail to correctly interpret or utilize the response. Enhancing the agent’s instruction/system prompt can help guide the model to use the MCP response more effectively.
- LLM behavior is often non-deterministic which can lead to varying results — sometimes good, sometimes incomplete or inaccurate.
- Occasionally, the MCP server returns errors (e.g., rate limits, API failures). If the MCP server isn’t functioning properly, check the JSON response for error messages. You can also use tools like Smithery (https://smithery.ai/) to verify whether MCP is working correctly.
- Running the LLM on a GPU instead of a CPU allows you to use more powerful models and obtain faster, more accurate responses.
- Testing remote MCP servers with different libraries (e.g., Google ADK, LangChain, PydanticAI) is also recommended to ensure broader compatibility and performance insights.