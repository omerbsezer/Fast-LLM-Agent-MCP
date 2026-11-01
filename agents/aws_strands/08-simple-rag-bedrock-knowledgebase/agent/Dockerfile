FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
# opentelemetry-instrument (ADOT) sends traces to AgentCore Observability.
CMD ["opentelemetry-instrument", "python3", "agent.py", "--serve"]
