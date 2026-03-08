# Fast LLM & Agents & MCPs
This repo covers LLM, Agents, MCP Tools, Skills concepts both theoretically and practically:
- LLM Architectures, RAG, Fine Tuning, Agents, Tools, MCP, Agent Frameworks, Reference Documents.
- Agent Sample Codes with **LangChain & LangGraph (> v1.0.0, prod. version)**.
- Agent Sample Codes with **AWS Strands Agents**.
- Agent Sample Codes with **Google Agent Development Kit (ADK)**.

# LangChain & LangGraph (v1.0.0) - Agent Sample Code & Projects
- [Sample-00: Basic Agent](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/00-basic-agent)
- [Sample-01: Agent With Static Tools](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/01-agent-with-static-tools)
- [Sample-02: Agent With Tools Structured Output](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/02-agent-with-tools-structured-output)
- [Sample-03: Agent Short Term Memory](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/03-agent-short-term-memory)
- [Sample-04: Agent Messages](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/04-agent-messages)
- [Sample-05: Agent PII Middleware Guardrail](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/05-agent-pii-middleware-guardrail)
- [Sample-06: Agent Subagents as Tool](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/06-agent-subagents-as-tool)
- [Sample-07: Agent Skills](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/07-agent-skills)
- [Sample-08: LangGraph Multi Agents Sequential Pattern](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/08-langgraph-multi-agents-sequential-pattern)
- [Sample-09: LangGraph Multi Agents Generator Evaluator Pattern](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/09-langgraph-multi-agents-generator-evaluator-pattern)
- [Sample-10: LangGprah Multi Agents Router Pattern](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/langchain_langgraph/10-langgraph-multi-agents-router-pattern)

# AWS Strands - Agent Sample Code & Projects
- [Sample-00: First Agent with AWS Strands Agents](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/aws_strands/00-first-agent-strands)
- [Sample-01: Agent Strands with Streamlit UI](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/aws_strands/01-agent-strands-function-streamlit)
- [Sample-02: Agent Strands Local MCP FileOps, Streamlit](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/aws_strands/02-agent-strands-local-mcp-fileOps-streamlit)
- [Sample-03: Agent Strands, Remote MCP Serper Google Search Tool with Streamlit](https://github.com/omerbsezer/Fast-LLM-Agent-MCP/tree/main/agents/aws_strands/03-agent-strands-remote-mcp-serper-streamlit)

# Google ADK - Agent Sample Code & Projects
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

# Table of Contents
- [Motivation](#motivation)
- [LLM Architecture & LLM Models](#llm)
- [Prompt Engineering](#promptengineering)
- [RAG: Retrieval-Augmented Generation](#rag)
- [Fine Tuning](#finetuning)
- [LLM Application Frameworks & Libraries](#llmframeworks)
  - [LangChain & LangGraph](#lang)
  - [LlamaIndex](#llamaindex)
- [Agent Frameworks](agentframework)
  - [LangChain & LangGraph Agents](#langagent) 
  - [AWS Strands Agents](#strands) 
  - [Google Agent Development Kit](#adk) 
  - [CrewAI Agents](#crewai)
  - [PydanticAI Agents](#pydanticai)  
- [Agents](#llmagents)
  - [Tools](#agenttools)
  - [MCP: Model Context Protocol](#mcp)
  - [Skills](#skills)
  - [A2A: Agent to Agent Protocol](#a2a)
- [References](#references)

## Motivation <a name="motivation"></a>
Why should we use / learn / develop LLM models and app? 

- 🔍 **Automate Complex Tasks:** Summarization, translation, coding, research, and content creation at scale.
- 🧠 **Enable Intelligent Apps:** Build chatbots, copilots, search engines, and data explorers with natural language interfaces.
- 🚀 **Boost Productivity:** Save time for users and businesses by handling repetitive or knowledge-based tasks.
- 💰 **High Market Demand:** Growing need for LLM apps in SaaS, enterprise tools, customer support, and education.
- 🧩 **Versatile Integration:** Combine with tools, APIs, databases, and agents for powerful workflows.
- 🛠️ **Customize for Edge Cases:** Fine-tune or adapt for domain-specific needs (finance, legal, health, etc.).
- 📈 **Career Growth:** Opens roles in AI engineering, MLOps, product innovation, and technical leadership.
- 🌍 **Global Impact:** Democratizes access to information and automation, especially for non-technical users.

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
- Embedding Models
  - OpenAI (text-embedding-ada-002), 
  - HuggingFace (all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2), 
  - AWS Titan embeddings
- Reranker Models  
- Generation Models
- Vector stores & Databases: 
  - FAISS, 
  - Pinecone, 
  - Weaviate, 
  - Chroma, 
  - Milvus, 
  - Qdrant


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

### LangChain & LangGraph <a name="lang"></a>
- LangChain is a framework for building LLM applications by combining models with tools, prompts, memory, and data sources. LangGraph is an extension of LangChain that models LLM workflows as stateful graphs, making it easier to build multi-step or multi-agent systems with loops and control flow.
- https://python.langchain.com/docs/introduction/

### LlamaIndex <a name="llamaindex"></a>
- LlamaIndex is a framework that helps connect large language models (like OpenAI GPT) to external data sources such as PDFs, databases, or APIs. It focuses on data ingestion, indexing, and retrieval so LLM apps (e.g., RAG systems) can query private or structured data efficiently
- https://developers.llamaindex.ai/python/framework/

## Agent Frameworks <a name="agentframework"></a>

Agent frameworks are specialized software tools or libraries designed to build, manage, and orchestrate LLM-based agents. These frameworks help you create intelligent agents that can autonomously reason, plan, and act using external tools or in coordination with other agents.

What Do Agent Frameworks Provide?
- **Agent Abstractions:** Define agent identity, goals, memory, behavior, and permissions.
- **Tool Interfaces:** Register external tools (APIs, functions, databases, etc.) agents can use.
- **Execution Logic:** Handle planning, decision-making, tool-calling, retries, and feedback loops.
- **Multi-Agent Orchestration:** Manage communication and task delegation among multiple agents.
- **Memory & Context:** Enable persistent memory, history tracking, and contextual reasoning.
- **Observability:** Offer tracing, logging, and step-by-step reasoning outputs for debugging.


### LangChain & LangGraph <a name="langagent"></a> 
- LangChain & LangGraph have also strong agent framework. LangChain provides a strong agent framework that lets LLMs dynamically decide which tools, APIs, or actions to use to complete a task. LangGraph extends this by enabling stateful, multi-agent workflows and complex control flows (loops, branching) using a graph-based execution model.
- https://python.langchain.com/docs/introduction/

### AWS Strands <a name="strands"></a>
- The Strands Agents SDK empowers developers to quickly build, manage, evaluate and deploy AI-powered agents.
- https://strandsagents.com/latest/


### Google Agent Development Kit <a name="adk"></a>
- Agent Development Kit (ADK) is a flexible and modular framework for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and is built for compatibility with other frameworks.
- ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.
- https://google.github.io/adk-docs/

### CrewAI <a name="crewai"></a>

CrewAI is  Python framework built entirely from scratch—completely independent of LangChain or other agent frameworks. CrewAI empowers developers with both high-level simplicity and precise low-level control, ideal for creating autonomous AI agents tailored to any scenario:

- **CrewAI Crews:** Optimize for autonomy and collaborative intelligence, enabling you to create AI teams where each agent has specific roles, tools, and goals.
- **CrewAI Flows:** Enable granular, event-driven control, single LLM calls for precise task orchestration and supports Crews natively.
- https://docs.crewai.com/introduction


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

```python
# Define a tool function
def get_capital_city(country: str) -> str:
  """Retrieves the capital city for a given country."""
  # Replace with actual logic (e.g., API call, database lookup)
  capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
  return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
    model="gemini-3.0-pro",
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

### Agent Skills <a name="skills"></a>
- Agent skills are predefined capabilities or tools that an AI agent can invoke to complete tasks (e.g., run code, query a database, or read documentation). They are usually packaged as reusable units containing instructions, scripts, and supporting files so the agent knows when and how to use them. In many frameworks, a skill may include a script (Python/JS) for execution and Markdown files describing usage instructions. This allows agents to extend functionality without changing the core agent logic.

Example skill structure:

```
skills/
 └─ query_parquet/
     ├─ skill.md
     └─ run_query.py
```

skill.md:
```
# Query Parquet Skill
Use this skill when you need to run SQL queries on Parquet files.

Input:
- SQL query

Output:
- Query results as a table
```

run_query.py:
```python
import duckdb

def run_query(sql, file_path):
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{file_path}') WHERE {sql}").fetchall()
```

- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview 

### A2A: Agent to Agent Protocol <a name="a2a"></a>

The Agent2Agent (A2A) protocol facilitates communication between independent AI agents. Here are the core concepts:
- **Agent Card:** A public metadata file (usually at /.well-known/agent.json) describing an agent's capabilities, skills, endpoint URL, and authentication requirements. Clients use this for discovery.
- **A2A Server:**  An agent exposing an HTTP endpoint that implements the A2A protocol methods (defined in the json specification). It receives requests and manages task execution.
- **A2A Client:**  An application or another agent that consumes A2A services. It sends requests (like tasks/send) to an A2A Server's URL.
- **Task:**  The central unit of work. A client initiates a task by sending a message (tasks/send or tasks/sendSubscribe). Tasks have unique IDs and progress through states (submitted, working, input-required, completed, failed, canceled).
- **Message:**  Represents communication turns between the client (role: "user") and the agent (role: "agent"). Messages contain Parts.
- **Part:**  The fundamental content unit within a Message or Artifact. Can be TextPart, FilePart (with inline bytes or a URI), or DataPart (for structured JSON, e.g., forms).
- **Artifact:**  Represents outputs generated by the agent during a task (e.g., generated files, final structured data). Artifacts also contain Parts.
- **Streaming:**  For long-running tasks, servers supporting the streaming capability can use tasks/sendSubscribe. The client receives Server-Sent Events (SSE) containing TaskStatusUpdateEvent or TaskArtifactUpdateEvent messages, providing real-time progress.
- **Push Notifications:**  Servers supporting pushNotifications can proactively send task updates to a client-provided webhook URL, configured via tasks/pushNotification/set.
- **Link:** https://github.com/google/A2A

## Other Useful Resources Related LLMs, Agents, MCPs <a name="other-resources"></a>
- https://www.promptingguide.ai/
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langgraph/overview
- https://strandsagents.com/latest/
- https://google.github.io/adk-docs/
- https://docs.crewai.com/
- https://modelcontextprotocol.io/
- https://smithery.ai/
- https://developer.nvidia.com/blog/category/generative-ai/
- https://huggingface.co/

## References <a name="references"></a>
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langgraph/overview
- https://strandsagents.com/latest/
- https://google.github.io/adk-docs/
- https://github.com/google/A2A