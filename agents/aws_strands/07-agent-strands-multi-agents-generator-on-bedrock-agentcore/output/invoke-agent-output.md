## Output Stdout

> ./deploy.sh   (it runs 'terraform apply')

...
Apply complete! Resources: 1 added, 0 changed, 0 destroyed.

Outputs:

agent_runtime_arn = "arn:aws:bedrock-agentcore:eu-central-1:xx:runtime/strands_blog_agentcore-YWpTH47VzV"
ecr_repository_url = "xx.dkr.ecr.eu-central-1.amazonaws.com/strands-blog-agentcore"
memory_id = "strands_blog_agentcore-LB7jGt9lCN"
runtime_execution_role_arn = "arn:aws:iam::xx:role/strands-blog-agentcore-runtime-role"
==> 4/4  Done.
agent_runtime_arn = "arn:aws:bedrock-agentcore:eu-central-1:xx:runtime/strands_blog_agentcore-YWpTH47VzV"
ecr_repository_url = "xx.dkr.ecr.eu-central-1.amazonaws.com/strands-blog-agentcore"
memory_id = "strands_blog_agentcore-LB7jGt9lCN"
runtime_execution_role_arn = "arn:aws:iam::xx:role/strands-blog-agentcore-runtime-role"

> export AGENT_RUNTIME_ARN=$(terraform -chdir=terraform output -raw agent_runtime_arn)

> python3 invoke_client.py "AI Agents with Memory on AWS Bedrock AgentCore"


{
  "topic": "AI Agents with Memory on AWS Bedrock AgentCore",
  "iterations": 1,
  "accepted": true,
  "blog": "# AI Agents with Memory on AWS Bedrock AgentCore\n\n## Introduction\n\nArtificial Intelligence (AI) agents are becoming increasingly sophisticated, capable of performing complex tasks and interactions. One of the key advancements in AI agent technology is the integration of memory capabilities. This allows agents to remember past interactions, user preferences, and contextual information, leading to more personalized and efficient experiences. AWS Bedrock AgentCore offers a robust platform for implementing AI agents with memory, providing a seamless way to build, deploy, and manage these intelligent systems.\n\n## What are AI Agents with Memory?\n\nAI agents with memory are designed to retain information from previous interactions, enabling them to provide more contextually aware and personalized responses. Unlike stateless agents, which treat each interaction as independent, memory-enabled agents can leverage past data to enhance their performance and user experience. This capability is particularly valuable in applications such as customer support, personalized recommendations, and dynamic content generation.\n\n## Benefits of AI Agents with Memory\n\n### Personalized User Experiences\n\nBy remembering user preferences and past interactions, memory-enabled agents can deliver highly personalized experiences. This leads to increased user satisfaction and engagement.\n\n### Improved Efficiency\n\nAgents with memory can avoid redundant information gathering, streamlining interactions and reducing the need for users to repeat themselves.\n\n### Contextual Awareness\n\nMemory allows agents to maintain context across sessions, making conversations more natural and coherent. This is especially useful in multi-turn dialogues where context is crucial.\n\n### Enhanced Learning\n\nAgents can learn from past interactions, improving their performance over time. This iterative learning process enables agents to become more accurate and effective.\n\n## Implementing AI Agents with Memory on AWS Bedrock AgentCore\n\n### Prerequisites\n\nTo get started with AWS Bedrock AgentCore, ensure you have the following:\n\n- An AWS account with appropriate permissions.\n- Python 3.10+ environment.\n- AWS CLI configured.\n\n### Setting Up AgentCore Memory\n\nAWS Bedrock AgentCore provides a comprehensive solution for building AI agents with memory. The process involves configuring memory strategies, deploying the agent, and testing its memory capabilities.\n\n#### Memory Configuration\n\nAgentCore Memory can be configured using the `AgentCoreMemoryConfig` class. This involves setting up retrieval strategies for user facts and preferences.\n\n```python\nfrom bedrock_agentcore.memory import AgentCoreMemoryConfig\n\nmemory_config = AgentCoreMemoryConfig(\n    user_preferences_strategy=\"semantic\",\n    session_summaries_strategy=\"episodic\"\n)\n```\n\n#### Deployment\n\nDeploy the memory-enabled agent using the `agentcore` CLI commands.\n\n```sh\nagentcore configure --memory-config memory_config.json\nagentcore launch\n```\n\n#### Testing\n\nTest the agent's memory capabilities by conducting short-term and long-term memory tests.\n\n```python\n# Short-term memory test within the same session\nagent.remember(\"user_preference\", \"likes_python\")\n\n# Long-term memory test across different sessions\nagent.remember(\"user_fact\", \"visited_website_last_month\")\n```\n\n## Memory Strategies in AgentCore\n\nAWS Bedrock AgentCore offers several built-in memory strategies to enhance agent performance:\n\n### User Preferences\n\nThis strategy stores and retrieves user preferences, allowing the agent to tailor responses based on individual user needs.\n\n### Semantic\n\nThe semantic strategy extracts key insights and facts from user interactions, enabling the agent to provide more informed responses.\n\n### Session Summaries\n\nSession summaries capture the essence of each interaction, helping the agent maintain context across sessions.\n\n### Episodic Memory\n\nEpisodic memory allows agents to learn from experiences and build knowledge over time, making them more adaptive and intelligent.\n\n## Best Practices for Implementing Memory in AI Agents\n\n### Collect Ground Truth Data\n\nGathering accurate ground truth data is essential for training and evaluating memory-enabled agents. This data helps agents learn effectively and provide reliable responses.\n\n### Use Bedrock Guardrails\n\nAWS Bedrock provides guardrails to ensure the safe and ethical use of AI agents. These guardrails help maintain high-quality interactions and prevent misuse.\n\n### Monitor with AgentCore Observability\n\nLeverage AgentCore Observability to monitor agent performance, track execution traces, and measure latency metrics. This helps identify areas for improvement and ensures optimal agent operation.\n\n### Break Monolithic Tools into Composable Pieces\n\nDesigning agents with smaller, composable tools enhances flexibility and testability. This approach allows for easier updates and maintenance.\n\n## Conclusion\n\nAI agents with memory represent a significant advancement in AI technology, offering personalized, efficient, and contextually aware interactions. AWS Bedrock AgentCore provides a powerful platform for implementing these memory-enabled agents, with robust memory strategies and deployment capabilities. By following best practices and leveraging the features of AgentCore, developers can create intelligent agents that deliver exceptional user experiences.\n\n## References\n\n1. [Bring AI agents with Long-Term memory into production in minutes - DEV Community](https://dev.to/aws/bring-ai-agents-with-long-term-memory-into-production-in-minutes-338l)\n2. [Building AI Agents on AWS in 2025: A Practitioner's Guide to Bedrock, AgentCore, and Beyond - DEV Community](https://dev.to/aws-builders/building-ai-agents-on-aws-in-2025-a-practitioners-guide-to-bedrock-agentcore-and-beyond-4efn)\n3. [AI Agent Memory Made Easy - Amazon Bedrock AgentCore Memory with Spring AI - DEV Community](https://dev.to/yuriybezsonov/ai-agent-memory-made-easy-amazon-bedrock-agentcore-memory-with-spring-ai-3bng)\n4. [Build Production-ready AI Agents with AWS Bedrock & Agentcore - DEV Community](https://dev.to/aws-builders/build-production-ready-ai-agents-with-aws-bedrock-agentcore-13kk)\n5. [Cracking the Bedrock, Reaching the Core: Building Agents with AWS AgentCore Runtime and Memory - DEV Community](https://dev.to/aws-builders/cracking-the-bedrock-reaching-the-core-building-agents-with-aws-agentcore-runtime-and-memory-32kg)\n6. [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)\n7. [AWS AgentCore Runtime Documentation](https://docs.aws.amazon.com/agentcore/)\n8. [Spring AI Amazon Bedrock AgentCore Starter Documentation](https://spring.io/projects/spring-ai)",
  "sources": [
    "https://dev.to/aws/bring-ai-agents-with-long-term-memory-into-production-in-minutes-338l",
    "https://dev.to/aws-builders/building-ai-agents-on-aws-in-2025-a-practitioners-guide-to-bedrock-agentcore-and-beyond-4efn",
    "https://dev.to/yuriybezsonov/ai-agent-memory-made-easy-amazon-bedrock-agentcore-memory-with-spring-ai-3bng",
    "https://dev.to/aws-builders/build-production-ready-ai-agents-with-aws-bedrock-agentcore-13kk",
    "https://dev.to/aws-builders/cracking-the-bedrock-reaching-the-core-building-agents-with-aws-agentcore-runtime-and-memory-32kg"
  ],
  "session_id": "7495615f-fb3d-4e89-9bf4-5abb161339fc"
}

> python3 invoke_client.py --read-memory 7495615f-fb3d-4e89-9bf4-5abb161339fc


# Memory — 

---


## Research

### Iter 1 — 14:46:59
**KW:** AI Agents with Memory on AWS Bedrock AgentCore, Implementing persistent memory for AWS Bedrock AgentCore, Memory management techniques for AWS Bedrock AI Agents, AWS Bedrock AgentCore memory integration best practices, Enhancing AWS Bedrock AgentCore with stateful memory, AWS Bedrock AgentCore memory storage solutions, Developing state-aware AI Agents on AWS Bedrock, AWS Bedrock AgentCore memory architecture overview

### [Bring AI agents with Long-Term memory into production in minutes - DEV Community](https://dev.to/aws/bring-ai-agents-with-long-term-memory-into-production-in-minutes-338l)
**Snippet:** April 15, 2026 - """ Production-Ready AI Agent with Memory Remembers conversations and user preferences across sessions """ import os from strands import Agent from strands_tools import calculator from bedrock_agentcore.runtime import BedrockAgentCoreApp from ...

**Key Points:**
- **Cross-Session Memory**: AgentCore Memory enables AI agents to remember users across different sessions, providing personalized experiences that persist beyond session boundaries.
- **Memory Architecture**: Combines AgentCore Runtime's built-in session memory with AgentCore Memory's cross-session persistence for long-term extraction of key insights and user preferences.
- **Prerequisites**: Requires an AWS account with appropriate permissions, Python 3.10+ environment, and AWS CLI configured.
- **Memory Configuration**: Involves setting up AgentCoreMemoryConfig with retrieval strategies for user facts and preferences.
- **Memory Strategies**: Offers built-in strategies like User Preferences, Semantic, and Session Summaries, with options for custom strategies.
- **Deployment**: Uses `agentcore configure` and `agentcore launch` commands to deploy memory-enabled agents, with custom headers for user identification.
- **Testing**: Includes short-term memory tests within the same session and long-term memory tests across different sessions, noting asynchronous extraction for long-term memory.

---
### [Building AI Agents on AWS in 2025: A Practitioner's Guide to Bedrock, AgentCore, and Beyond - DEV Community](https://dev.to/aws-builders/building-ai-agents-on-aws-in-2025-a-practitioners-guide-to-bedrock-agentcore-and-beyond-4efn)
**Snippet:** January 3, 2026 - The December update added episodic memory - agents that learn from experiences and build knowledge over time. Instead of treating each session as independent, the agent develops understanding of patterns and preferences. from bedrock_agentcore ...

**Key Points:**
- AWS is shifting from simple LLM invocations to orchestrating autonomous AI agents that plan, execute, learn, and operate independently.
- Amazon Bedrock serves as the multi-model foundation, offering nearly 100 serverless foundation models and supporting reinforcement fine-tuning and cross-region inference.
- Amazon Bedrock AgentCore provides a full stack for building, deploying, and operating agents, featuring session isolation, long-running workloads, and bidirectional streaming.
- AgentCore Memory enables agents to remember context across interactions through episodic memory, allowing them to learn from experiences and build knowledge over time.
- AgentCore Gateway converts existing APIs into Model Context Protocol (MCP) compatible tools, enabling dynamic tool discovery for multi-agent systems.
- AgentCore Identity handles authentication and authorization for agent actions, supporting OAuth integration and secure vault storage for credentials.
- AgentCore Observability integrates with CloudWatch for end-to-end monitoring, including execution traces, latency metrics, and custom dashboards.
- Policy and Evaluations in AgentCore provide guardrails for production deployment, enabling real-time interception of tool calls and built-in evaluators for quality dimensions.

---
### [AI Agent Memory Made Easy - Amazon Bedrock AgentCore Memory with Spring AI - DEV Community](https://dev.to/yuriybezsonov/ai-agent-memory-made-easy-amazon-bedrock-agentcore-memory-with-spring-ai-3bng)
**Snippet:** February 3, 2026 - But if you're already on AWS and want to move fast, Amazon Bedrock AgentCore Memory handles the infrastructure so you can focus on what your agent actually does. ... Let's create an Amazon Bedrock AgentCore Memory resource with LTM strategies. Copy and run in terminal: # Create memory resource MEMORY_ID=$(aws bedrock-agentcore-control create-memory \ --name "memory_demo" --description "Memory for AI Agent demo" \ --event-expiry-duration 90 --no-cli-pager \ --query "memory.id" --output text) && echo "Memory ID: ${MEMORY_ID}" # Wait for memory to become active echo -n "Waiting for memory to beco

**Key Points:**
- Amazon Bedrock AgentCore Memory provides fully managed, persistent memory for AI agents with both short-term and long-term memory capabilities.
- Short-term memory (STM) stores raw conversation events within sessions, maintaining context for multi-turn conversations.
- Long-term memory (LTM) strategies include semantic extraction, user preferences, summaries, and episodic memory, automatically extracting facts and preferences asynchronously.
- AgentCore Memory eliminates the need for custom infrastructure like PostgreSQL, offering built-in strategies and automatic extraction.
- The Spring AI Amazon Bedrock AgentCore starter simplifies integration, requiring minimal code (under 50 lines of Java) and no manual database management.
- Prerequisites include Java 21+, AWS CLI with Bedrock AgentCore access, and JBang.
- The memory resource creation involves AWS CLI commands to set up and activate memory with specified strategies.
- The AI agent application uses Spring Boot and Spring AI dependencies, with environment variables configuring memory IDs and strategy IDs.
- Testing the application demonstrates the agent’s ability to remember user information across sessions using both STM and LTM.

---
### [Build Production-ready AI Agents with AWS Bedrock & Agentcore - DEV Community](https://dev.to/aws-builders/build-production-ready-ai-agents-with-aws-bedrock-agentcore-13kk)
**Snippet:** January 26, 2026 - Most AI chatbots are stateless. They forget everything after each conversation. But for a SaaS product, we need persistence. We need the agent to remember: ... from bedrock_agentcore.memory import MemoryClient from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent, AfterInvocationEvent class ProductHuntMemoryHooks(HookProvider): def __init__(self, memory_id: str, client: MemoryClient, actor_id: str, session_id: str): self.memory_id = memory_id self.client = client self.actor_id = actor_id self.session_id = session_id def retrieve_product_context(self, event: MessageAddedEvent)

**Key Points:**
- Utilizes AWS Bedrock for accessing foundation models like Claude.
- Employs Strands Agents SDK for building AI agents with minimal code.
- Uses AgentCore Runtime for session isolation and security in production.
- Implements AgentCore Memory for persistent context across sessions with short-term and long-term memory.
- Builds custom tools using the @tool decorator in Strands for agent capabilities.
- Integrates FastAPI with Server-Sent Events (SSE) for streaming responses in the web interface.
- Recommends Infrastructure as Code (IaC) for consistent and manageable deployment.
- Suggests breaking monolithic tools into smaller, composable pieces for flexibility and testability.
- Plans for multi-agent systems to scale SaaS applications.
- Emphasizes best practices like collecting ground truth data, using Bedrock Guardrails, and monitoring with AgentCore Observability.

---
### [Cracking the Bedrock, Reaching the Core: Building Agents with AWS AgentCore Runtime and Memory - DEV Community](https://dev.to/aws-builders/cracking-the-bedrock-reaching-the-core-building-agents-with-aws-agentcore-runtime-and-memory-32kg)
**Snippet:** May 8, 2026 - Bedrock serves every LLM call via Strands' BedrockModel. AgentCore Memory receives session events automatically from the Strands session manager, and returns extracted patterns when the loop asks for them at the start of each run.

**Key Points:**
- AWS AgentCore Runtime handles HTTP transport, session lifecycle, and streaming framing.
- AgentCore allows integration with various frameworks like Strands, LangGraph, and CrewAI.
- The solution consists of an imperative shell (agent/builder.py) and a functional core (agent/iteration.py).
- AgentCore Memory automatically stores and retrieves session events for cross-session learning.
- The generate/score/critique/regenerate loop runs in the functional core, independent of the framework.
- AWS Bedrock serves LLM calls, while AgentCore Memory manages session-based learning and pattern extraction.
- Observability is achieved through OpenTelemetry spans and structured logging in the imperative shell.

## Sources

### Iter 1
- https://dev.to/aws/bring-ai-agents-with-long-term-memory-into-production-in-minutes-338l
- https://dev.to/aws-builders/building-ai-agents-on-aws-in-2025-a-practitioners-guide-to-bedrock-agentcore-and-beyond-4efn
- https://dev.to/yuriybezsonov/ai-agent-memory-made-easy-amazon-bedrock-agentcore-memory-with-spring-ai-3bng
- https://dev.to/aws-builders/build-production-ready-ai-agents-with-aws-bedrock-agentcore-13kk
- https://dev.to/aws-builders/cracking-the-bedrock-reaching-the-core-building-agents-with-aws-agentcore-runtime-and-memory-32kg

## Critiques

### Iter 1 — 14:47:14 — **ACCEPTED**
| Depth | Recency | Structure | Writing |
|---|---|---|---|
| 5/5 | 5/5 | 5/5 | 5/5 |

## Log

- **Eval iter 1** `14:47:14` — ACCEPTED (D:5 R:5 S:5 W:5)
- **Iter 1** `14:46:59` — 5 articles

> /terraform/ terraform destroy

...
Do you really want to destroy all resources?
  Terraform will destroy all your managed infrastructure, as shown above.
  There is no undo. Only 'yes' will be accepted to confirm.

  Enter a value: yes

aws_iam_role_policy.runtime_permissions: Destroying... [id=strands-blog-agentcore-runtime-role:strands-blog-agentcore-runtime-permissions]
aws_bedrockagentcore_agent_runtime.agent[0]: Destroying...
aws_iam_role_policy.runtime_permissions: Destruction complete after 0s
aws_ecr_repository.agent: Destroying... [id=strands-blog-agentcore]
aws_ecr_repository.agent: Destruction complete after 0s
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [00m10s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [00m20s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [00m30s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [00m40s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [00m50s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [01m00s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [01m10s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [01m20s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [01m30s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [01m40s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [01m50s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [02m00s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [02m10s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [02m20s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [02m30s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [02m40s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [02m50s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Still destroying... [03m00s elapsed]
aws_bedrockagentcore_agent_runtime.agent[0]: Destruction complete after 3m7s
aws_iam_role.runtime_execution: Destroying... [id=strands-blog-agentcore-runtime-role]
aws_bedrockagentcore_memory.agent: Destroying... [id=strands_blog_agentcore-LB7jGt9lCN]
aws_iam_role.runtime_execution: Destruction complete after 2s
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 00m10s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 00m20s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 00m30s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 00m40s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 00m50s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 01m00s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 01m10s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 01m20s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 01m30s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 01m40s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 01m50s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 02m00s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 02m10s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 02m20s elapsed]
aws_bedrockagentcore_memory.agent: Still destroying... [id=strands_blog_agentcore-LB7jGt9lCN, 02m30s elapsed]
aws_bedrockagentcore_memory.agent: Destruction complete after 2m36s

Destroy complete! Resources: 5 destroyed.