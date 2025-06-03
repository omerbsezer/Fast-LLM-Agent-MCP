# Fast LLM & Agents & MCPs
This repo covers LLM concepts both theoretically and practically:
- LLM Architectures, RAG, Fine Tuning, Agents, Tools, MCP, Agent Frameworks, Reference Documents.
  
# Table of Contents
- [Motivation](#motivation)
- [LLM Architecture & LLM Models](#llm)
- [Prompt Engineering](#promptengineering)
- [RAG: Retrieval-Augmented Generation](#rag)
- [Fine Tuning](#finetuning)
- [LLM Application Frameworks & Libraries](#llmframeworks)
   - [LangChain](#langchain)
   - [LangGraph](#langgraph)
   - [LLamaIndex](#llamaindex)
   - [PydanticAI](#pydanticai)
- [Agents](#llmagents)
  - [Tools](#agenttools)
  - [MCP: Model Context Protocol](#mcp)
  - [A2A: Agent to Agent Protocol](#a2a)
- [Agent Frameworks](agentframework)
  - [Google Agent Development Kit](#adk) 
  - [CrewAI](#crewai)
  - [PraisonAI Agents](#praisonai) 
- [Samples Projects](#samples)
  - [Project1: AI Content Detector with AWS Bedrock, Llama 3.1 405B](#ai-content-detector)
  - [Project2: LLM with Model Context Protocol (MCP) using PraisonAI, Ollama, LLama 3.1 1B,8B](#localllm-mcp-praisonai)
  - [Project3: Agent with Google ADK and ADK Web](#agent-adk-web)
  - [Project4: Agent Container with Google ADK, FastAPI, Streamlit](#agent-adk-container-streamlit)
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
   - LlamaIndex: https://www.llamaindex.ai/

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


## Agents <a name="llmagents"></a>

### Tools <a name="agenttools"></a>

### MCP: Model Context Protocol <a name="mcp"></a>
https://www.anthropic.com/engineering/building-effective-agents

## Agent Frameworks <a name="agentframework"></a>


## Sample Projects <a name="samples"></a>

### Project1: AI Content Detector with AWS Bedrock, Llama 3.1 405B <a name="ai-content-detector"></a>

### Project2: LLM with Model Context Protocol (MCP) using PraisonAI, Ollama, LLama 3.1 1B,8B <a name="localllm-mcp-praisonai"></a>

### Project3: Agent with Google ADK and ADK Web <a name="agent-adk-web"></a>

### Project4: Agent Container with Google ADK, FastAPI, Streamlit <a name="agent-adk-container-streamlit"></a>

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

