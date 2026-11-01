import streamlit as st

from utils.bedrock_ops import get_ingestion_job_status, query, start_ingestion_job
from utils.s3_ops import upload_file

st.set_page_config(page_title="Simple RAG", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_job_id" not in st.session_state:
    st.session_state.ingestion_job_id = None

with st.sidebar:
    st.header("Chat History")
    for turn in st.session_state.chat_history:
        st.markdown(f"**{turn['role'].capitalize()}:** {turn['content']}")

st.title("Simple RAG")

st.subheader("Upload documents")
uploaded_files = st.file_uploader("Choose file(s)", accept_multiple_files=True)
if uploaded_files and st.button("Upload to S3"):
    for uploaded_file in uploaded_files:
        try:
            key = upload_file(uploaded_file, uploaded_file.name)
            st.success(f"Uploaded '{key}' to S3.")
        except Exception as e:
            st.error(f"Upload failed for '{uploaded_file.name}': {e}")

st.subheader("Sync Knowledge Base")
if st.button("Sync Knowledge Base"):
    try:
        st.session_state.ingestion_job_id = start_ingestion_job()
        st.info(f"Ingestion job started: {st.session_state.ingestion_job_id}")
    except Exception as e:
        st.error(f"Sync failed: {e}")

if st.session_state.ingestion_job_id:
    try:
        job_info = get_ingestion_job_status(st.session_state.ingestion_job_id)
        st.write(f"Last ingestion job status: **{job_info['status']}**")
        if job_info["status"] == "FAILED" and job_info["failure_reasons"]:
            st.error("Ingestion failed: " + "; ".join(job_info["failure_reasons"]))
    except Exception as e:
        st.error(f"Could not fetch job status: {e}")

st.subheader("Ask a question")
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        result = query(question)
        answer = result["answer"]
        if result["citations"]:
            answer += "\n\nSources:\n" + "\n".join(f"- {c}" for c in result["citations"])
        else:
            answer += "\n\n_No source documents were found for this question — make sure you've uploaded and synced documents._"
    except Exception as e:
        answer = f"Error: {e}"
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
