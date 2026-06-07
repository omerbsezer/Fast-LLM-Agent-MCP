# Memory — AI Agents with Memory on AWS Bedrock AgentCore  (2026-06-07 15:48)

---

## Research

### Iter 1 — 15:48:54
**KW:** AI Agents with Memory on AWS Bedrock AgentCore, Benefits of AI Agents with Memory using AWS Bedrock AgentCore, Implementing AI Agents with Memory on AWS Bedrock AgentCore, Best practices for AI Agents with Memory on AWS Bedrock AgentCore, Comparing AI Agents with Memory on AWS Bedrock AgentCore vs other platforms, Case studies of AI Agents with Memory on AWS Bedrock AgentCore, Security considerations for AI Agents with Memory on AWS Bedrock AgentCore, Future trends for AI Agents with Memory on AWS Bedrock AgentCore

### [Bring AI agents with Long-Term memory into... - DEV Community](https://dev.to/aws/bring-ai-agents-with-long-term-memory-into-production-in-minutes-338l)
**Snippet:** Amazon Bedrock AgentCore Memory solves this. Your agents remember users across sessions, learn from past interactions, and provide personalized experiences that persist beyond session boundaries. This tutorial shows you how to add long-term memory to your agents.

**Key Points:**
- AgentCore Runtime provides built-in short-term memory within sessions (up to 8 hours or 15 minutes of inactivity).
- AgentCore Memory adds cross-session persistence, allowing agents to remember users and their preferences across different sessions.
- AgentCore Memory automatically extracts and stores key insights and user preferences for long-term retention.
- Three built-in memory strategies: User Preferences, Semantic, and Session Summaries.
- Custom headers (e.g., X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id) are used to pass user identifiers for user-centric memory storage.
- Long-term memory extraction is an asynchronous process, requiring several minutes between storing and retrieving memories.
- The tutorial demonstrates setting up, configuring, and testing an AI agent with cross-session memory on AWS Bedrock AgentCore.

---
### [Build Production AI Agents with Managed Long-Term Memory](https://dev.to/aws/build-production-ai-agents-with-managed-long-term-memory-2jm)
**Snippet:** Amazon Bedrock AgentCore Memory provides a managed alternative. This service handles cross-session persistence, embedding generation, storage, and optimization automatically. You configure memory strategies through API calls instead of building custom infrastructure.

**Key Points:**
- Amazon Bedrock AgentCore Memory replaces custom vector storage with fully managed long-term memory for AI agents.
- AgentCore handles cross-session persistence, embedding generation, storage, and optimization automatically.
- Memory strategies are configured via API calls instead of custom infrastructure.
- Prerequisites include an AWS account with Bedrock and AgentCore access, Python 3.10+, and AWS CLI configured.
- Required packages: `strands-agents`, `bedrock-agentcore`, `aws-opentelemetry-distro`, and `boto3`.
- AgentCore CLI (`agentcore configure`) automatically creates memory resources and sets environment variables.
- Memory stores user preferences, dietary restrictions, budget considerations, past travel experiences, and user context.
- Memory client and storage are created using `bedrock_agentcore.memory.MemoryClient` and configured with `AgentCoreMemoryConfig`.
- Agent entry point (`invoke` function) extracts session and actor IDs, creates/retrieves agent instances with memory, and processes user messages.
- Multi-modal inputs (images and videos) are sent as base64-encoded payloads and processed by specific tools (`image_reader` and `video_reader_local`).
- Agent deployment to production involves configuring memory strategies and launching with `agentcore launch`.
- Testing includes short-term memory within sessions, long-term memory across sessions, and image/video processing.

---
### [AWS Bedrock AgentCore Memory: Give Your AI Agent a Brain That...](https://dev.to/sampathkaran/aws-bedrock-agentcore-memory-give-your-ai-agent-a-brain-that-actually-remembers-12ie)
**Snippet:** Tagged with agents, ai, aws, llm.AgentCore Memory is AWS's answer to this. It's a managed memory service purpose-built for agents, with three distinct memory tiers and a retrieval API that plugs directly into the Bedrock agent runtime. Let's actually use it.

**Key Points:**
- AWS Bedrock AgentCore Memory addresses the limitations of stateless agents by providing a managed memory service.
- Three memory tiers: Session Memory (in-context working memory), Long-Term (Semantic) Memory (stores facts from past conversations), and Episodic Memory (stores sequences of events).
- Setup involves creating a memory store with specified memory types and storage duration.
- Memory consolidation filters control what data is promoted from session to long-term memory, reducing noise and cost.
- Deleting memory records is crucial for compliance with data deletion requests (GDPR).
- Two architecture patterns: per-user memory stores for isolation or shared stores with namespaced session IDs for simplicity.
- Observability features allow tracing memory retrievals to debug agent behavior.
- Cost considerations include storage and retrieval fees, with options to manage costs through configuration settings.

---
### [AI Agent Memory Made Easy - Amazon Bedrock AgentCore...](https://dev.to/yuriybezsonov/ai-agent-memory-made-easy-amazon-bedrock-agentcore-memory-with-spring-ai-3bng)
**Snippet:** Tagged with agents, ai, aws, java.Why Amazon Bedrock AgentCore Memory? AI models are stateless - ask "What's my name?" after introducing yourself, and the model has no idea. AgentCore Memory solves this with: Fully Managed: AWS handles storage, extraction, and retention.

**Key Points:**
- Amazon Bedrock AgentCore Memory provides fully managed short-term and long-term memory for AI agents, handling storage, extraction, and retention.
- Short-term memory (STM) stores raw conversation events within sessions, maintaining context for multi-turn conversations.
- Long-term memory (LTM) strategies include semantic extraction of factual information, user preferences identification, conversation summarization, and episodic capture of meaningful interaction slices.
- AgentCore Memory eliminates the need for custom infrastructure like PostgreSQL, reducing code from ~300 lines to ~50 lines.
- The Spring AI Amazon Bedrock AgentCore starter integrates with AgentCore Memory, enabling easy setup and use in Java applications.
- Prerequisites include Java 21+, AWS CLI with Bedrock AgentCore access, and JBang.
- The demo application uses JBang to create a Spring Boot AI agent with memory in under 50 lines of Java.
- Environment variables configure the memory resource and strategies for the agent.
- The agent demonstrates persistent memory by remembering user facts across sessions and extracting semantic facts asynchronously.

---
### [AI Agent Memory: Manual, Mem0, LangMem, & AWS AgentCore](https://dev.to/sudarshangouda/ai-agent-memory-from-manual-implementation-to-mem0-to-aws-agentcore-2d7c)
**Snippet:** Amazon Bedrock AgentCore Memory is a fully managed service from AWS designed to give agents state and memory without managing infrastructure. What Makes It Special? Unlike Mem0 or manual solutions where you manage the database, AWS handles everything.

**Key Points:**
- **Manual Memory Implementation**: Utilizes pure Python with JSON files for persistence and a Priority Queue for working memory.
- **LangMem**: A library for long-term memory management, allowing agents to decide what to remember using memory tools.
- **ChromaDB**: A vector database for semantic memory, using OpenAI embeddings for semantic search over knowledge.
- **Mem0 & Supabase**: Dual storage solution for production apps; Mem0 handles memory extraction and storage, while Supabase provides vector and SQL databases.
- **AWS Bedrock AgentCore**: Mentioned as a service for implementing AI agents with memory, though specific technical details are not provided in the article.

## Sources

### Iter 1
- https://dev.to/aws/bring-ai-agents-with-long-term-memory-into-production-in-minutes-338l
- https://dev.to/aws/build-production-ai-agents-with-managed-long-term-memory-2jm
- https://dev.to/sampathkaran/aws-bedrock-agentcore-memory-give-your-ai-agent-a-brain-that-actually-remembers-12ie
- https://dev.to/yuriybezsonov/ai-agent-memory-made-easy-amazon-bedrock-agentcore-memory-with-spring-ai-3bng
- https://dev.to/sudarshangouda/ai-agent-memory-from-manual-implementation-to-mem0-to-aws-agentcore-2d7c

## Critiques

### Iter 1 — 15:49:15 — **ACCEPTED**
| Depth | Recency | Structure | Writing |
|---|---|---|---|
| 5/5 | 5/5 | 5/5 | 5/5 |

## Log

- **Eval iter 1** `15:49:15` — ACCEPTED (D:5 R:5 S:5 W:5)

- **Iter 1** `15:48:54` — 5 articles
