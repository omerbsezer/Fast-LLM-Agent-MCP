import boto3
import os
import json

MODEL_ID = 'meta.llama3-1-405b-instruct-v1:0'  
AWS_REGION = 'us-west-2'  

class AIGeneratedAnalyzer:
    def __init__(self):
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'
            # you can use also Environment Variable to give access keys, or use .aws/credentials
            # aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            # aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

    def analyze_text(self, text):
        # prompt engineering/template => asking specific way to get correctly the patterns 
        prompt = f"""Analyze this text for AI generation indicators. Provide:
            1. AI Score (0-100)
            2. Clear conclusion
            3. Specific technical patterns with explanations
            
            Text: {text}
            
            Format response EXACTLY like:
            ## AI Score: [score]/100
            ## Conclusion: [conclusion]
            ## Patterns:
            - [Pattern Name]: [Description] (Confidence: High/Medium/Low)
            - [Pattern Name]: [Description] (Confidence: High/Medium/Low)
            ..."""
         
        # context window 2048, temperature 0.2   
        response = self.bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 2048,
                "temperature": 0.2
            })
        )
        return self._parse_response(response)

    def _parse_response(self, response):
        # get back result as json format
        result = json.loads(response.get('body').read()).get('generation')
        
        analysis = {
            'score': None,
            'conclusion': None,
            'patterns': []
        }

        # parse the response => create analysis report in dictionary format 
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('## AI Score:'):
                analysis['score'] = int(line.split(':')[-1].split('/')[0].strip())
            elif line.startswith('## Conclusion:'):
                analysis['conclusion'] = line.split(':')[-1].strip()
            elif line.startswith('-'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    pattern_name = parts[0][2:].strip()  # Remove leading '-'
                    description = parts[1].split('(Confidence:')[0].strip()
                    confidence = parts[1].split('Confidence:')[-1].replace(')', '').strip()
                    analysis['patterns'].append({
                        'pattern': pattern_name,
                        'description': description,
                        'confidence': confidence
                    })

        return analysis