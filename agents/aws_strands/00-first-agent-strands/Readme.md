## First Agent with AWS Strands Agents
It shows first agent implementation with AWS Strands Agents.

Please run Python files on Linux, or WSL on Win.

Install AWS Strands Agents:
- pip install strands-agents
- pip install strands-agents-tools strands-agents-builder


### AWS Bedrock Model 
- First, to enable in your region or AWS-West for Model Access (AWS Bedrock > Bedrock Configuration > Model Access > Nova Pro, or Claude 3.7 Sonnet, or Llama 4)
- In these samples, we'll use AWS Nova Pro, because it's served in different regions by AWS. After model access, give permission to your IAM to access AWS Bedrock services. 
- 2 Options to reach AWS Bedrock Model using your AWS Account:

#### 1. AWS Config
- With 'aws configure', to create 'config' and 'credentials' files

#### 2. Getting variables using .env file
Add .env file:

``` 
AWS_ACCESS_KEY_ID= PASTE_YOUR_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY=PASTE_YOUR_SECRET_ACCESS_KEY_HERE
``` 

### Demo
Run:

``` 
cd ./Fast-LLM/agents/aws_strands/00-first-agent
python3 -u my_agent/agent.py
``` 


### Reference
https://strandsagents.com/