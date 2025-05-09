# Fast LLM & Agents & MCPs
This repo covers LLM, Agents, MCP Tools concepts both theoretically and practically:
- LLM Architectures, RAG, Fine Tuning, Agents, Tools, MCP, Agent Frameworks, Reference Documents.
- Agent Sample Codes with Google Agent Development Kit (ADK).

# Agent Sample Code & Projects
- [Sample-00: Agent with Google ADK and ADK Web](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/00-first-agent-with-adk-web)
- [Sample-01: Agent Container with Google ADK, FastAPI, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/01-agent-function-tools-container-streamlit)
- [Sample-02: Agent Local MCP Tool (FileServer) with Google ADK, FastAPI, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/02-agent-local-mcp-fileOps-streamlit)
- [Sample-03: Agent Remote MCP Tool (Web Search: Serper) with Google ADK, FastAPI, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/03-agent-remote-mcp-google-search-serper)
- [Sample-04: Agent Memory and Builtin Google Search Tool with Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/04-agent-memory-builtin-search-tool)
- [Sample-05: Agent LiteLLM - AWS Bedrock (Llama3.1-405B), Ollama with Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/05-agent-litellm-bedrock-ollama)
- [Sample-06: Multi-Agent Sequential, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/06-multiagent-workflow-sequential)
- [Sample-07: Multi-Agent Parallel, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/07-multiagent-workflow-parallel)
- [Sample-08: Multi-Agent Loop, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/08-multiagent-loop)
- [Sample-09: Multi-Agent Hierarchy, Streamlit GUI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/09-multiagent-hierarchy)

# LLM Projects
- [Project1: AI Content Detector with AWS Bedrock, Llama 3.1 405B](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/projects/Project1-AI-Content-Detector-AWS-Bedrock)
- [Project2: LLM with Model Context Protocol (MCP) using PraisonAI, Ollama, LLama 3.1 1B,8B](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/projects/Project2-MCP-Agent-Ollama)

# Table of Contents
- [Motivation](#motivation)
- [LLM Architecture & LLM Models](#llm)
- [Prompt Engineering](#promptengineering)
- [RAG: Retrieval-Augmented Generation](#rag)
- [Fine Tuning](#finetuning)
- [LLM Application Frameworks & Libraries](#llmframeworks)
  - [LangChain-LangGraph](#lang)
- [Agent Frameworks](agentframework)
  - [Google Agent Development Kit](#adk) 
  - [CrewAI](#crewai)
  - [PraisonAI Agents](#praisonai) 
  - [PydanticAI](#pydanticai)  
- [Agents](#llmagents)
  - [Tools](#agenttools)
  - [MCP: Model Context Protocol](#mcp)
  - [A2A: Agent to Agent Protocol](#a2a)
- [Agent Samples](#samples)
  - [Sample-00: Agent with Google ADK and ADK Web](#agent-adk-web)
  - [Sample-01: Agent Container with Google ADK, FastAPI, Streamlit GUI](#agent-adk-container-streamlit)
  - [Sample-02: Agent Local MCP Tool (FileServer) with Google ADK, FastAPI, Streamlit GUI](#agent-local-mcp-fileOps-streamlit)
  - [Sample-03: Agent Remote MCP Tool (Web Search: Serper) with Google ADK, FastAPI, Streamlit GUI](#agent-remote-mcp-google-search-serper)
  - [Sample-04: Agent Memory and Builtin Google Search Tool with Streamlit GUI](#agent-memory-builtin-search-tool)
  - [Sample-05: Agent LiteLLM - AWS Bedrock (Llama3.1-405B), Ollama with Streamlit GUI](#agent-litellm-bedrock-ollama)
  - [Sample-06: Multi-Agent Sequential, Streamlit GUI](#multi-agent-sequential)
  - [Sample-07: Multi-Agent Parallel, Streamlit GUI](#multi-agent-parallel)
  - [Sample-08: Multi-Agent Loop, Streamlit GUI](#multi-agent-loop)
  - [Sample-09: Multi-Agent Hierarchy, Streamlit GUI](#multi-agent-hierarchy)
 - [LLM Projects](#llm-projects)
  - [Project1: AI Content Detector with AWS Bedrock, Llama 3.1 405B](#ai-content-detector)
  - [Project2: LLM with Model Context Protocol (MCP) using PraisonAI, Ollama, LLama 3.1 1B,8B](#localllm-mcp-praisonai)
- [Other Useful Resources Related LLMs, Agents, MCPs](#other-resources)
- [References](#references)

## Motivation <a name="motivation"></a>
Why should we use/learn/develop LLM models and app? 

## LLM Architecture & LLM Models <a name="llm"></a>
- An LLM (Large Language Model) is a type of artificial intelligence model trained on massive amounts of text data to understand and generate human-like language.
- "Large": Refers to the billions of parameters and the vast datasets used during training.
- "Language Model": Predicts and generates text, based on the context of the input.
- LLMs are built on **Transformer architecture** (uses self-attention to understand context)
- LLMs are **decoder-only** transformers.

LLM Core components:
- **Embedding layer:** Converts words/tokens into vectors.
- **Positional encoding:** Adds word order information.
- **Self-attention mechanism:** Captures relationships between tokens.
- **Feed-forward layers:** Process token-level information.
- **Layer normalization & residuals:** Stabilize and enhance training.
- **Output head:** predicts next token or classification result.

## Prompt Engineering <a name="promptengineering"></a>
- Prompt engineering is the practice of designing and refining input prompts to get the best possible responses from LLMs (Large Language Models).
- Crafting clear, specific instructions.
- Using structured formats (e.g., lists, JSON, examples).
- Applying techniques like:
  - Zero-shot prompting (just ask)
  - Few-shot prompting (give examples)
  - Chain-of-thought prompting (ask the model to explain step by step)
- Why Is It Important?
  - **Improves output quality:** A well-crafted prompt guides the LLM to produce relevant and accurate results.
  - **Boosts model accuracy:** Helps LLMs follow complex instructions. Precise instructions reduce false or made-up information.
  - **Enables task-specific performance:** You can get the model to summarize, translate, generate code, or follow workflows with just a well-written prompt.
  - **Cost-efficient:** Improves results without retraining or fine-tuning large models.
  - **Critical for GenAI apps:** Critical for building chatbots, agents, search assistants, and copilots.
  - **Supports RAG and Tool Use:** Helps orchestrate LLMs with tools, APIs, or knowledge bases in advanced systems.

## RAG: Retrieval-Augmented Generation <a name="rag"></a>
RAG (Retrieval-Augmented Generation) is a technique that combines external knowledge retrieval with LLM-based text generation.
How It Works:
- **Retrieve:** Search a knowledge base (e.g., documents, PDFs, databases) using a query. Typically uses vector search with embeddings (e.g., FAISS, Elasticsearch).
- **Augment:** Inject the retrieved text into the prompt.
- **Generate:** The LLM uses this context to produce a more accurate, grounded response.

Why RAG Is Important:
- **Reduces hallucinations** by grounding LLM responses in real data.
- **Keeps answers up to date** (LLMs don’t need retraining for every update).
- **Improves trust** and transparency (can show sources).
- **Scalable for enterprise:** Used in chatbots, search, document QA, agents.

RAG Popular/Common Tools:
- Embedding models: https://python.langchain.com/docs/integrations/text_embedding/
  - OpenAI (text-embedding-ada-002), 
  - HuggingFace (all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2), 
  - AWS Titan embeddings
- Vector stores: https://python.langchain.com/docs/integrations/vectorstores/ 
  - FAISS, 
  - Pinecone, 
  - Weaviate, 
  - Chroma, 
  - Milvus, 
  - ElasticsearchStore,
  - Redis,
- Frameworks: 
   - LangChain: https://python.langchain.com/docs/introduction/

## Fine Tuning <a name="finetuning"></a>
Fine-tuning is the process of adapting a pre-trained LLM model  to a specific task or domain by further training it on a smaller, specialized dataset. The goal is to make the model better at handling particular requirements without starting from scratch.

How Fine-Tuning Works:
- **Start with a Pre-trained Model:** A large model (Llama3.3, Mistral) that has been trained on vast amounts of general data (pdfs) to understand language.
- **Use Domain-Specific Data:** Provide a smaller dataset that is specific to the task or domain you care about (e.g., legal text, medical data, customer service).
- **Train the Model:** The model’s parameters are updated with this new, specialized data to make it perform better in the target domain, but it retains the general knowledge learned from the larger training dataset.
- **Apply the Fine-Tuned Model:** The model is now able to generate or classify text based on the specialized domain, often achieving much higher performance than a general model.

Why Fine-Tuning Is Important:
- **Improves Accuracy:** The model becomes more precise in your specific use case, like medical text analysis or customer support chatbots.
- **Saves Time and Resources:** You don’t have to train a new model from scratch (which requires vast computing resources and data).
- **Customizes Behavior:** Fine-tuning allows you to adjust a model to behave in a way that aligns with your application’s goals (e.g., tone, context, style).

Fine-Tuning Methods:
- **Full Fine-Tuning:** Fine-tuning **all parameters** of a pre-trained model for a specific task. It offers maximum customization but is resource-intensive and requires large datasets
- **Feature-based Fine-Tuning:** Only the **final layers of a pre-trained model** are fine-tuned, with the rest of the model frozen. It’s faster and less resource-heavy than full fine-tuning but may not provide the same level of task-specific optimization.
- **PFET (Progressive Fine-Tuning)**: Fine-tuning occurs in **stages**, starting with simpler tasks and progressing to more complex ones. This method reduces catastrophic forgetting and improves the model’s ability to learn complex tasks gradually but is more time-consuming.
- **Adapter Fine-Tuning**: Small **trainable layers** (adapters) are added to a pre-trained model, and only these adapters are fine-tuned. It’s highly efficient for multi-task learning but less flexible than full fine-tuning and may not be optimal for all tasks.
- **LoRA (Low-Rank Adaptation):** Low-rank matrices are **added to specific layers** of a pre-trained model, and only these matrices are fine-tuned. It’s efficient in terms of memory and computation, making it ideal for large models but may not be as effective for very complex tasks.
- **QLoRA (Quantized LoRA):** Combines LoRA with quantization techniques to further **reduce the size of the model** and its memory requirements. The low-rank matrices are quantized, making them more memory-efficient.

## LLM Application Frameworks & Libraries <a name="llmframeworks"></a>

LLM (Large Language Model) Application Frameworks and Libraries are tools designed to simplify the development, orchestration, and deployment of applications powered by large language models like GPT, Claude, Gemini, or LLaMA. These tools provide abstractions for managing prompts, memory, agents, tools, workflows, and integrations with external data or systems.

### LangChain-LangGraph <a name="lang"></a>
- LangChain implements a standard interface for large language models and related technologies, such as embedding models and vector stores, and integrates with hundreds of providers.
- https://python.langchain.com/docs/introduction/


## Agent Frameworks <a name="agentframework"></a>

Agent frameworks are specialized software tools or libraries designed to build, manage, and orchestrate LLM-based agents. These frameworks help you create intelligent agents that can autonomously reason, plan, and act using external tools or in coordination with other agents.

What Do Agent Frameworks Provide?
- **Agent Abstractions:** Define agent identity, goals, memory, behavior, and permissions.
- **Tool Interfaces:** Register external tools (APIs, functions, databases, etc.) agents can use.
- **Execution Logic:** Handle planning, decision-making, tool-calling, retries, and feedback loops.
- **Multi-Agent Orchestration:** Manage communication and task delegation among multiple agents.
- **Memory & Context:** Enable persistent memory, history tracking, and contextual reasoning.
- **Observability:** Offer tracing, logging, and step-by-step reasoning outputs for debugging.

### Google Agent Development Kit <a name="adk"></a>

- Agent Development Kit (ADK) is a flexible and modular framework for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and is built for compatibility with other frameworks.
- ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.
- https://google.github.io/adk-docs/

### CrewAI <a name="crewai"></a>

CrewAI is  Python framework built entirely from scratch—completely independent of LangChain or other agent frameworks. CrewAI empowers developers with both high-level simplicity and precise low-level control, ideal for creating autonomous AI agents tailored to any scenario:

- CrewAI Crews: Optimize for autonomy and collaborative intelligence, enabling you to create AI teams where each agent has specific roles, tools, and goals.
- CrewAI Flows: Enable granular, event-driven control, single LLM calls for precise task orchestration and supports Crews natively.
- https://docs.crewai.com/introduction

### PraisonAI Agents <a name="praisonai"></a> 

- PraisonAI is a production-ready Multi-AI Agents framework with self-reflection, designed to create AI Agents to automate and solve problems ranging from simple tasks to complex challenges.
- https://docs.praison.ai/

### PydanticAI <a name="pydanticai"></a> 

- PydanticAI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.
https://ai.pydantic.dev/
  
## Agents <a name="llmagents"></a>

An LLM agent is an autonomous or semi-autonomous system that uses a large language model (like GPT, Claude, or Gemini) to reason, make decisions, and take actions using external tools or APIs to accomplish a goal.

An LLM agent typically includes:
- **LLM Core:** The brain that interprets tasks and generates reasoning.
- **Memory (optional):** Stores history of actions, inputs, and outputs for context-aware behavior.
- **Tools:** External APIs, databases, web search, code execution, or custom functions the agent can call.
- **Environment:** The runtime or framework (e.g., LangChain, CrewAI, Google ADK) in which the agent operates.

```
# Define a tool function
def get_capital_city(country: str) -> str:
  """Retrieves the capital city for a given country."""
  # Replace with actual logic (e.g., API call, database lookup)
  capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
  return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country... (previous instruction text)""",
    tools=[get_capital_city]
)
```

### Tools <a name="agenttools"></a>

A Tool represents a specific capability provided to an AI agent, enabling it to perform actions and interact with the world beyond its core text generation and reasoning abilities.

- **Function Tool:** Transforming a function into a tool is a straightforward way to integrate custom logic into your agents.
- **Agent-as-Tool:** This powerful feature allows you to leverage the capabilities of other agents within your system by calling them as tools. The Agent-as-a-Tool enables you to invoke another agent to perform a specific task, effectively delegating responsibility.
- **Google ADK Built-in-Tools:** These built-in tools provide ready-to-use functionality such as Google Search or code executors that provide agents with common capabilities. 
- **Langchain Tools:** Also used in Google ADK as Langchain Tools.
  - https://python.langchain.com/docs/integrations/tools/
- **CrewAI Tools:** Also used in Google ADK as CrewAI Tools.
  - https://docs.crewai.com/concepts/tools
- **MCP Tools:** Connects MCP server apps to MCP Clients (Claude App, VSCode, etc,) or Agents (Google ADK).
  - MCP Servers: https://github.com/modelcontextprotocol/servers
  
### MCP: Model Context Protocol <a name="mcp"></a>
- The Model Context Protocol (MCP) is an open standard designed to standardize how Large Language Models (LLMs) like Gemini and Claude communicate with external applications, data sources, and tools
- MCP follows a client-server architecture, defining how data (resources), interactive templates (prompts), and actionable functions (tools) are exposed by an MCP server and consumed by an MCP client (which could be an LLM host application or an AI agent).
- https://www.anthropic.com/engineering/building-effective-agents

## Agent Samples 

### Sample-00: Agent with Google ADK and ADK Web <a name="agent-adk-web"></a>
It shows first agent implementation with Google UI ADK Web and Gemini API key.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/00-first-agent-with-adk-web

![sample-00](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/00-first-agent-with-adk-web/gif/weather-time-agent-adk.gif)

### Sample-01: Agent Container with Google ADK, FastAPI, Streamlit GUI <a name="agent-adk-container-streamlit"></a>
Customized front-end app with streamlit and dockerized backend-app using FastAPI to communicate both Gemini LLM and front-end.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/01-agent-function-tools-container-streamlit

![sample-01](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/01-agent-function-tools-container-streamlit/gif/weather-time-agent-container-streamlit.gif)

### Sample-02: Agent Local MCP Tool (FileServer) with Google ADK, FastAPI, Streamlit GUI  <a name="agent-local-mcp-fileOps-streamlit"></a>
It shows how to run local MCP tool with Gemini.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/02-agent-local-mcp-fileOps-streamlit

![sample-02](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/02-agent-local-mcp-fileOps-streamlit/gif/agent-local-mcp-streamlit.gif)

### Sample-03: Agent Remote MCP Tool (Web Search: Serper) with Google ADK, FastAPI, Streamlit GUI  <a name="agent-remote-mcp-google-search-serper"></a>
It shows how to run remote MCP tool with Gemini.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/03-agent-remote-mcp-google-search-serper

![sample-03](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/03-agent-remote-mcp-google-search-serper/gif/agent-remote-mcp-google-search-serper.gif)

### Sample-04: Agent Memory and Builtin Google Search Tool with Streamlit GUI  <a name="agent-memory-builtin-search-tool"></a>
It shows how to run built-in tool Google Search and Agent Memory.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/04-agent-memory-builtin-search-tool

**With Memory:**
![sample-04-with-memory](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/04-agent-memory-builtin-search-tool/gif/agent-with-memory.gif)


**Without Memory:**
![sample-04-without-memory](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/04-agent-memory-builtin-search-tool/gif/agent-without-memory.gif)

### Sample-05: Agent LiteLLM - AWS Bedrock (Llama3.1-405B), Ollama with Streamlit GUI  <a name="agent-litellm-bedrock-ollama"></a>
It shows how to connect Llama 3.1 405B on AWS Bedrock and Ollama Llama3.2:1b running on local PC using Litellm.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/05-agent-litellm-bedrock-ollama

![sample-05](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/05-agent-litellm-bedrock-ollama/gif/agent-bedrock-llama3.1-405.gif)

### Sample-06: Multi-Agent Sequential, Streamlit GUI  <a name="multi-agent-sequential"></a>
It shows how to implement multi-agent sequential workflow.

- **Link:**  https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/06-multiagent-workflow-sequential

![sample-06](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/06-multiagent-workflow-sequential/gif/multi-agent-sequential.gif)

### Sample-07: Multi-Agent Parallel, Streamlit GUI  <a name="multi-agent-parallel"></a>
It shows how to implement multi-agent parallel workflow.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/07-multiagent-workflow-parallel

![sample-07](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/07-multiagent-workflow-parallel/gif/multi-agent-parallel-merger.gif)

### Sample-08: Multi-Agent Loop, Streamlit GUI  <a name="multi-agent-loop"></a>
It shows how to implement multi-agent loop workflow.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/08-multiagent-loop

![sample-08](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/agents/google_adk/08-multiagent-loop/gif/multi-agent-loop-code-checker.gif)

### Sample-09: Multi-Agent Hierarchy, Streamlit GUI  <a name="multi-agent-hierarchy"></a>
It shows how to implement multi-agent hierarchy runs.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/google_adk/09-multiagent-hierarchy


## LLM Projects <a name="llm-projects"></a>

### Project-01: AI Content Detector with AWS Bedrock, Llama 3.1 405B <a name="ai-content-detector"></a>
It shows how to implement AI Content Detector with AWS Bedrock, Llama 3.1 405B and prompt templating power.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/projects/Project1-AI-Content-Detector-AWS-Bedrock

![project-01](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/projects/Project1-AI-Content-Detector-AWS-Bedrock/gif/text-devto-human-analysis.gif)

### Project-02: LLM with Model Context Protocol (MCP) using PraisonAI, Ollama, LLama 3.2:1B and LLama 3.1:8B <a name="localllm-mcp-praisonai"></a>
It shows how to implement MCP remote tools using PraisonAI, Ollama LLama 3.2:1B and LLama 3.1:8B.

- **Link:** https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/projects/Project2-MCP-Agent-Ollama

![project-02](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/blob/main/projects/Project2-MCP-Agent-Ollama/gif/serper-ask.gif)

## Other Useful Resources Related LLMs, Agents, MCPs <a name="other-resources"></a>
- https://www.promptingguide.ai/
- https://python.langchain.com/docs/introduction/
- https://docs.crewai.com/
- https://docs.praison.ai/
- https://modelcontextprotocol.io/
- https://smithery.ai/
- https://developer.nvidia.com/blog/category/generative-ai/
- https://huggingface.co/

## References <a name="references"></a>
- https://google.github.io/adk-docs/
