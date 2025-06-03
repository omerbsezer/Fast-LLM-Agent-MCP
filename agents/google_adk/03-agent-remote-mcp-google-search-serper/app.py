import streamlit as st
import requests

st.set_page_config(page_title="Agent Chat", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Agent Remote Web Search MCP")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask to search for real-time data or anything...")

# send and display user + assistant messages
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"query": user_query}
        )
        response.raise_for_status()
        agent_reply = response.json().get("response", "No response.")
    except Exception as e:
        agent_reply = f"Error: {str(e)}"

    st.chat_message("assistant").markdown(agent_reply)
    st.session_state.messages.append({"role": "assistant", "content": agent_reply})
