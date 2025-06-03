import streamlit as st
import requests

st.set_page_config(page_title="Agent Chat", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Multi-Agent Sequential")

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
        # respond is not str, it is list, because of multiple agent responds.
        all_replies = response.json().get("responses", ["No response."])

        for reply in all_replies:
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.chat_message("assistant").markdown(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

