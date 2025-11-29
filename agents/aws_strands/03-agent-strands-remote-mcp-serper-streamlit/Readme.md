## Agent Strands, Remote MCP Serper Google Search Tool with Streamlit
It shows remote MCP Serper Google Search tool with AWS Strands Agents.

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

#### Alternative Way
- 'aws configure' command configures your credential and config. If not, you can add them manually.
- In your Win "C:\Users\<username>\.aws" or in your Linux "/home/<username>/.aws", add 'config' and 'credentials' files 

- credentials:
``` 
[default]
aws_access_key_id = PASTE_YOUR_ACCESS_KEY_ID_HERE
aws_secret_access_key = PASTE_YOUR_SECRET_ACCESS_KEY_HERE
``` 

- config:
``` 
[default]
region = us-west-2
output = json
``` 

### Run Agent

Please run non-root username. 

```
uvicorn agent:app --host 0.0.0.0 --port 8000
```


### GUI
After running container, then pls run streamlit, it asks to htttp://localhost:8000/ask

```
streamlit run app.py
or
python -m streamlit run app.py
```

### Prompts

```
- I want to search "what is AWS Strands Agents?" on Google using Serper.
```

### Reference
https://strandsagents.com/