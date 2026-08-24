## AWS-Strands Multi Agents Generator-Evaluator Pattern

It shows AWS Strands multi agents patterns.

```
Multi-agent blog pipeline with AWS Strands SDK.

Architecture
────────────
  PlannerAgent: decides what to do next (plan / replan dynamically)
       - calls @tool: generate_blog      →  GeneratorAgent  (research + write)
       - calls @tool: evaluate_blog      →  EvaluatorAgent  (quality gate)
       - calls @tool: summarize_article  →  SummarizerAgent (key-point extraction)
       - calls @tool: search_web         →  DDGS search
       - calls @tool: fetch_page         →  HTTP scraper (full text, no truncation)
       - calls @tool: read_memory        →  AWS Bedrock AgentCore Memory
       - calls @tool: write_memory       →  AWS Bedrock AgentCore Memory

The key difference from pipeline:
  - The Planner's reasoning loop (via Strands agent loop) selects tools at runtime; it can re-search, summarise extra sources, rewrite, skip evaluation, or stop early.
  - GeneratorAgent and EvaluatorAgent are themselves Strands Agents exposed as @tool callables (agents-as-tools pattern).
  - SummarizerAgent is a dedicated low-temperature agent that fetches the full page and distills 5-10 key technical bullets per article before passing them to the writer.
  - No fixed edges: the Planner decides whether to call evaluate_blog or generate_blog next based on EvalResult feedback and iteration count.
```

Please run Python files on Linux, or WSL on Win.

### Enabling Virtual Environment
Virtual Env (venv):

``` 
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "✅ Virtual environment ready. Activate it with:"
echo "   source .venv/bin/activate"
``` 

To exit from venv:

``` 
deactivate
``` 

Requirements include latest version => lang-chain > 1.0.0; not 0.3 version.

### AWS Bedrock Model 
- First, to enable in your region or AWS-West for Model Access (AWS Bedrock > Bedrock Configuration > Model Access > Nova Pro, or Claude, or other open source models)
- 2 Options to reach AWS Bedrock Model using your AWS Account:

#### 1. AWS Config
- With 'aws configure', to create 'config' and 'credentials' files

#### 2. Getting variables using .env file
##### 2.1. Using Access Key ID 
Add .env file:

``` 
AWS_ACCESS_KEY_ID= PASTE_YOUR_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY=PASTE_YOUR_SECRET_ACCESS_KEY_HERE
``` 

##### 2.2. Using Token From Bedrock
You can create long and short time on Bedrock, then add both .env, or add environment variables.

.env:
```
AWS_BEARER_TOKEN_BEDROCK=XXX
```

or bash:
```bash
export AWS_BEARER_TOKEN_BEDROCK=XXX
```


### AWS Bedrock Open Source Models with OpenAI Compatible API 

List all possible models on that region with OpenAI Compatible API:
```
curl -X GET https://bedrock-mantle.eu-central-1.api.aws/v1/models -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" | jq
```

List:
```
{
  "data": [
    {
      "created": 17,
      "id": "qwen.qwen3-vl-235b-a22b-instruct",
      "object": "model",
      "owned_by": "system"
    },
    {
      "created": 17,
      "id": "nvidia.nemotron-nano-12b-v2",
      "object": "model",
      "owned_by": "system"
    },
    {
      "created": 17,
      "id": "openai.gpt-oss-120b",
      "object": "model",
      "owned_by": "system"
    },
    ....
  ]
}
```

### Running on AWS Bedrock AgentCore

Unlike the previous lab (`06-...generator-evaluator-pattern`), which keeps its research/critique memory in a local Markdown file, **this lab keeps no local memory at all**: both the agents and their memory run on AWS Bedrock AgentCore.
- **Agents** run as a container on **AWS Bedrock AgentCore Runtime**.
- **Memory** (Research/Sources/Critiques/Log) lives in **AWS Bedrock AgentCore Memory** (short-term/event store, no local file, ever).

Infrastructure is provisioned with Terraform (`terraform/`), scoped to `eu-central-1` by default (change `var.aws_region` to move it).

Note this is a separate concern from where the *models* run: `us.amazon.nova-pro-v1:0` is a cross-region inference profile only invokable from US Bedrock regions, regardless of `var.aws_region`. `agent.py` pins model calls to `BEDROCK_REGION` (env var, defaults to `us-east-1`), independent of the AgentCore infra region — override it if you swap to a model available elsewhere.

What gets deployed:
| File | Role |
|---|---|
| `agent.py` | Pipeline + both entrypoints: `python3 agent.py [topic]` runs it as a local process (still reads/writes the *deployed* AgentCore Memory — see below); `python3 agent.py --serve` runs it as the AgentCore Runtime service (`BedrockAgentCoreApp`) |
| `Dockerfile` | ARM64 image (required by AgentCore Runtime), serves `:8080` via `agent.py --serve` |
| `terraform/` | ECR repo, IAM execution role, AgentCore Memory resource, AgentCore Runtime resource |
| `deploy.sh` | Runs the 3 steps below in order |
| `invoke_client.py` | boto3 client to invoke the deployed Runtime (kept separate from `agent.py` — it only needs `boto3`, not the full agent dependency stack) |

`AgentCoreMemory` in `agent.py` is the only memory implementation: it writes Research/Sources/Critiques/Log as AgentCore Memory events, scoped to a fresh
`session_id` per run (memory never carries over between runs, only within one). It requires `AGENTCORE_MEMORY_ID` to be set — there is no local-file
fallback, so **the Terraform stack must be applied before any run, local or deployed.**

Note the credentials split: a local `python3 agent.py` process still uses **your** AWS credentials (`.env` / `aws configure`, as above) to call Bedrock
models *and* AgentCore Memory directly. Once deployed, the container instead runs under the **IAM execution role Terraform creates** — no `.env` is baked into the image.

#### Prerequisites
- AWS credentials with Bedrock / AgentCore / ECR / IAM permissions (`aws configure`, or `AWS_PROFILE`)
- Terraform >= 1.6
- Docker with `buildx` (for the `linux/arm64` cross-build)

#### Easiest path: deploy everything and invoke it remotely (recommended)
`deploy.sh` is the one command you need — it runs `terraform init`/`apply` (ECR + IAM + Memory), builds & pushes the image, then applies again to create the Runtime. Don't run `terraform apply` by hand first; the script already does it.

```bash
./deploy.sh
export AGENT_RUNTIME_ARN=$(terraform -chdir=terraform output -raw agent_runtime_arn)
python3 invoke_client.py "AI Agents with Memory on AWS Bedrock AgentCore"
```

#### Alternative: quick local dev loop
Skips the container/Runtime entirely — the agent runs as a plain local process, but still talks to the *same* AgentCore Memory. This path does **not** use `deploy.sh`, so it needs its own (smaller) manual apply, just for the Memory resource:

```bash
terraform -chdir=terraform init
terraform -chdir=terraform apply    # creates ECR repo, IAM role, AgentCore Memory (no Runtime yet)
export AGENTCORE_MEMORY_ID=$(terraform -chdir=terraform output -raw memory_id)
python3 agent.py "AI Agents with Memory on AWS Bedrock AgentCore"
```

Either path prints/returns a `session_id` when it finishes — save it to inspect that run's memory afterwards.

#### View AgentCore Memory content
There is no local file to `cat` — memory only exists in AgentCore. If you deployed via `deploy.sh`, invoke the same Runtime with a `read_memory` action instead of a topic (this reads through the Runtime's own execution role, so it works even if your local AWS credentials don't have AgentCore Memory permissions directly):

```bash
python3 invoke_client.py --read-memory <session_id>
```
Running `python3 agent.py` locally (the dev-loop alternative) already reads/writes the same AgentCore Memory resource directly with your own credentials — no extra step needed there, `read_memory`/`write_memory` are tools the Planner calls during the run itself.

#### Tear down
```bash
terraform -chdir=terraform destroy
```

### Reference
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langgraph/overview
