## Agent Container and Streamlit UI

## Container
Before creating image on agent, pls add .env with Gemini API Keys

``` 
# agent/.env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
``` 

Create image and container:

```
docker build -t weather-agent . 
docker run -d -p 8000:8000 --name weather-agent weather-agent
``` 

To stop, remove:
```
docker container rm -f weather-agent
docker ps -a
docker images
docker image rm weather-agent
``` 


## GUI
After running container, then pls run streamlit, it asks to htttp://localhost:8000/ask

```
streamlit run app.py
```

## CLI

``` 
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "What is the weather in New York?"}'
{"response":"The weather in New York is sunny with a temperature of 25°C (77°F)."}
``` 


## With Container (Optional, for Debug)
You can also run and debug without creating container. Run on the different terminal:

``` 
# ./agent
uvicorn main:app --host 0.0.0.0 --port 8000
``` 