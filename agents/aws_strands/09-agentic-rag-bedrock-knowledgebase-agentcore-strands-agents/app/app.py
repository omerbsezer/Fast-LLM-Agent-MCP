import sys
import uuid

import streamlit as st
from streamlit import runtime

# `python app.py` re-launches itself under Streamlit, so `streamlit run` isn't needed.
if __name__ == "__main__" and not runtime.exists():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())

from utils.agentcore_ops import is_agentic_configured, query_agentic
from utils.bedrock_ops import get_ingestion_job_status, query, start_ingestion_job
from utils.s3_ops import upload_file

st.set_page_config(page_title="Simple RAG", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_job_id" not in st.session_state:
    st.session_state.ingestion_job_id = None
if "agent_session_id" not in st.session_state:
    # AgentCore requires runtimeSessionId of at least 33 characters (a UUID is 36).
    st.session_state.agent_session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Chat History")
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f"**User:** {turn['content']}")
        else:
            st.markdown(f"**Assistant:** {turn['simple']['answer']}")

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
agentic_available = is_agentic_configured()
compare = st.checkbox(
    "Also run Agentic RAG (compare)",
    disabled=not agentic_available,
    help=None
    if agentic_available
    else "Run ./deploy.sh to deploy the agent on AgentCore; it writes AGENT_RUNTIME_ARN into app/.env.",
)

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        simple_result = query(question)
    except Exception as e:
        simple_result = {"answer": f"Error: {e}", "citations": []}

    agentic_result = None
    if compare:
        try:
            agentic_result = query_agentic(question, st.session_state.agent_session_id)
        except Exception as e:
            agentic_result = {"answer": f"Error: {e}", "citations": [], "trace": []}

    st.session_state.chat_history.append(
        {"role": "assistant", "simple": simple_result, "agentic": agentic_result}
    )


def render_answer(result: dict) -> None:
    st.markdown(result["answer"])
    if result.get("citations"):
        st.markdown("**Sources:**\n" + "\n".join(f"- {c}" for c in result["citations"]))
    else:
        st.caption("No source documents were found for this question.")


def render_trace(trace: list) -> None:
    if not trace:
        return
    with st.expander(f"Retrieval trace ({sum(1 for t in trace if t['type'] == 'retrieve')} retrieval calls)"):
        for step in trace:
            if step["type"] == "retrieve":
                reranked = " (reranked)" if step.get("reranked") else ""
                st.markdown(f"- **Retrieve** `{step['query']}` → {step['num_chunks']} chunks{reranked}")
            else:
                verdict = "sufficient" if step.get("sufficient") else "insufficient"
                st.markdown(f"- **Critique** → {verdict}, missing: {step.get('missing_aspects') or 'none'}")


def render_agentcore_proof(info: dict | None) -> None:
    if not info:
        return
    container = info.get("container") or {}
    echoed = container.get("session_id") == info["session_id"]
    with st.expander(f"Ran on Bedrock AgentCore {'✅' if echoed else '⚠️'}"):
        st.markdown(
            f"- **Runtime:** `{info['runtime_arn'].rsplit('/', 1)[-1]}`\n"
            f"- **Session ID:** `{info['session_id']}`\n"
            f"- **Trace ID:** `{info.get('trace_id') or 'n/a'}`\n"
            f"- **AWS request ID:** `{info['request_id']}` (HTTP {info.get('status_code')}, {info['latency_s']} s)\n"
            f"- **Container host:** `{container.get('host', 'n/a')}`, session seen inside the container: "
            f"`{container.get('session_id', 'n/a')}` {'(matches)' if echoed else '(does not match)'}"
        )
        st.markdown(
            f"Find this session under **Sessions** in "
            f"[AgentCore Observability]({info['console_url']}) to see each prompt, tool call "
            "and model call. Traces can take a few minutes to appear."
        )


for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
        continue

    with st.chat_message("assistant"):
        if turn["agentic"] is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Simple RAG")
                render_answer(turn["simple"])
            with col2:
                st.markdown("#### Agentic RAG")
                render_answer(turn["agentic"])
                render_trace(turn["agentic"].get("trace", []))
                render_agentcore_proof(turn["agentic"].get("agentcore"))
        else:
            render_answer(turn["simple"])
