import streamlit as st
from agents import route_query
import multiprocessing

def main():
    st.title("MCP Ollama Agent App")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to search?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Processing..."):
            response = route_query(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for safe subprocess creation on some platforms
    main()