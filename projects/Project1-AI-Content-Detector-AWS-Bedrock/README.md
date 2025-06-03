# AI-Content-Detector
Tool using AWS Bedrock Service - Llama 3.1 405B
- to give **AI generated score**, 
- to **analyze and explain** how much input text is AI generated.

## Motivation
- I created post in **Dev.to** and discuss how to eliminate COMPLETELY AI Generated Posts. 
  - https://dev.to/omerberatsezer/why-should-we-keep-writing-lets-discuss-2nc
- After discussion in the post, I decided to create a simple SAMPLE AI Content Detector tool with **PROMPT TEMPLATE** to show that it's really **SIMPLE and EFFECTIVE**.
- I mentioned this app in the following DEV.to post, please have a look for details, and please leave your valuable ideas in the post's comment:
  - https://dev.to/omerberatsezer/open-source-ai-content-detector-app-with-aws-bedrock-to-separate-completely-ai-vs-human-generated-2de4 

## Structure
- **ai_generated_analyzer.py** => prompt template, parse response
- **main.py** => Streamlit GUI
  - left side => Input Text
  - right side => AI Generation Score, Evaluation, Detection Patterns
    
## Run
- Before run, please be sure that your AWS credential and config files are correctly configured on **C:\Users\USERNAME\\.aws\config** and **C:\Users\USERNAME\\.aws\credentials**. Also, getting permission request in your AWS account is required to use Llama 3.1 405B on AWS Bedrock. Code does not offer to use AWS Bedrock Llama 3.1 405B for free, you'll use your own AWS account.

- Please download AWS CLI on your PC, then configure with your credentials to connect your AWS account.
  - https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html 
```shell
aws config
```

- **C:\Users\USERNAME\\.aws\config**
```yaml
[default]
region = us-west-2
output = json
```

- **C:\Users\USERNAME\\.aws\credentials**
```yaml
[default]
aws_access_key_id = AKIXXXXXXXXXXXX
aws_secret_access_key = XXXXXXXXXXXXX
```

- Another way, instead of using config/credential files, use your credentials with Environment Variables or copy to credentials to in your code as hardcoded (not suggested in the long term, but in short term, you can run the code).
  
- Run:
```shell
git clone https://github.com/omerbsezer/AI-Content-Detector
pip install -r requirements.txt
python -m streamlit run .\main.py
```

### AI Score for Human Generated Text
- My Post Analysis => https://dev.to/omerberatsezer/why-should-we-keep-writing-lets-discuss-2nc

![text-devto-human-analysis](https://github.com/omerbsezer/AI-Content-Detector/blob/main/gif/text-devto-human-analysis.gif)

### AI Score for COMPLETELY AI Generated Text
![text1-analysis](https://github.com/omerbsezer/AI-Content-Detector/blob/main/gif/text1-analysis.gif)

## Tests
- Sample text under the test-input-files directory

## Detected Patterns - Analysis
### AI Score for Human Generated Text in App
![mydevto-post-score-analysis](https://github.com/omerbsezer/AI-Content-Detector/blob/main/gif/mydevto-post-score-analysis.png)

### AI Score for COMPLETELY AI Generated Text
![image](https://github.com/user-attachments/assets/0eab6388-46e8-4869-aa22-4bbe4edf8715)

![image](https://github.com/user-attachments/assets/963c9fb0-2e75-499a-bc62-39f9f28079db)

