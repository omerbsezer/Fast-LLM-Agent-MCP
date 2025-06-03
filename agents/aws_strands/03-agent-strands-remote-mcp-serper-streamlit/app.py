import streamlit as st
import requests
import json 

st.set_page_config(page_title="AWS Strands Agent Chat", layout="centered")

st.title("AWS Strands Remote MCP Serper")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

user_query = st.chat_input("Ask for anything...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    try:
        response = requests.post("http://localhost:8000/ask", json={"query": user_query})
        response.raise_for_status()

        response_json = response.json()
        assistant_text = response_json["response"]["message"]["content"][0]["text"]

    except Exception as e:
        assistant_text = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

# display all messages in order
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])    