## LangGraph Multi Agents Generator-Evaluator Pattern

It shows first agent implementation with Langchain.

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

### Demo
Run:

``` 
python agent.py
``` 


### Reference
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langgraph/overview
