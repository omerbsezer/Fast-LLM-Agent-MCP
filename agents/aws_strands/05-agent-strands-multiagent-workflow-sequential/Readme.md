## Agent Strands with Streamlit UI
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

### Sample Prompt / Output 

```
Prompt: I want to go to Maldives, give me information about Maldives?

Welcome to your comprehensive travel guide for the Maldives! We're excited to help you begin your journey to this stunning paradise. Below, you'll find everything you need to know about the must-see attractions, historical highlights, where to stay, and culinary delights that await you in the Maldives. 

### Must-See Attractions
1. **Male' National Mosque** - The largest mosque in the Maldives, renowned for its breathtaking architecture and tranquil ambiance.
2. **HP Reef** - A world-class dive site celebrated for its rich marine biodiversity and vibrant coral reefs.
3. **Kurumba Beach** - A stunning beach with pristine white sands and crystal-clear waters, perfect for relaxation and various water sports.
4. **Sultan Park** - A serene park in Male' featuring a replica of the Islamic Centre and a peaceful pond, offering a glimpse into the local culture.
5. **Sea Turtle Excursion** - Numerous islands provide opportunities to observe sea turtles in their natural habitat, a must-do for wildlife enthusiasts.

### Historical Highlights
1. **Buddhist History** - Before embracing Islam in the 12th century, the Maldives were a Buddhist nation, reflecting a rich cultural heritage.
2. **Sultanate Era** - The Maldives were governed by a sultanate for centuries, a period that shaped the nation's history and traditions.
3. **Independence from the UK** - The Maldives achieved independence from the United Kingdom on July 26, 1965, marking a significant milestone in its history.
4. **Tsunami of 2004** - The 2004 Indian Ocean tsunami had a profound impact on the Maldives, leading to extensive reconstruction and resilience efforts.

### Where to Stay
1. **Male'** - The vibrant capital city, offering a deep dive into local culture and convenient access to major attractions.
2. **North Male' Atoll** - Renowned for its luxurious resorts and exceptional diving opportunities, ideal for a lavish getaway.
3. **Vaavu Atoll** - A serene and less crowded atoll, perfect for travelers seeking a more intimate and authentic Maldivian experience.

### Culinary Delights
1. **Garudhiya** - A traditional fish soup, often enjoyed during special occasions and celebrations.
2. **Mas Huni** - A popular breakfast dish made from tuna, coconut, and onions, reflecting the Maldivian love for seafood.
3. **Heka** - A delightful Maldivian snack made from rice flour and coconut, offering a taste of local flavors.

### Suggested Web Pages
To further enhance your trip research, we recommend visiting the following websites:
- [Visit Maldives Official Tourism Website](https://visitmaldives.com/)
- [Lonely Planet - Maldives Travel Guide](https://www.lonelyplanet.com/maldives)
- [TripAdvisor - Maldives](https://www.tripadvisor.com/Tourism-g293955-Maldives-Vacations.html)
- [National Geographic - Maldives](https://www.nationalgeographic.com/travel/destination/maldives)

We hope this guide helps you get started on planning your dream trip to the Maldives. If you have any more questions or need further assistance, please feel free to reach out. Safe travels and enjoy your adventure!
```


### Reference
https://strandsagents.com/