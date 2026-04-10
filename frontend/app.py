import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ─────────────────────────────────────────────
# Sidebar — document upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Text-based PDFs only — scanned documents are not supported"
    )

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_docs:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    )
                    data = response.json()

                    if response.status_code == 200:
                        st.session_state.uploaded_docs.append(uploaded_file.name)
                        if data.get("skipped"):
                            st.info(f"{uploaded_file.name} already indexed")
                        else:
                            st.success(f"Indexed {data['chunks']} chunks from {uploaded_file.name}")
                    else:
                        st.error(f"Upload failed: {data.get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is the server running?")

    st.divider()

    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs = response.json()["documents"]
            if docs:
                st.markdown("**Indexed documents**")
                for doc in docs:
                    st.markdown(f"- {doc}")
            else:
                st.caption("No documents indexed yet")
    except requests.exceptions.ConnectionError:
        st.caption("API offline")

    st.divider()

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

# ─────────────────────────────────────────────
# Main — chat interface
# ─────────────────────────────────────────────
st.title("RAG Chatbot")
st.caption("Ask questions about your uploaded documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['filename']}** — page {source['page']} "
                        f"*(relevance: {source['score']})*"
                    )

if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "history":  st.session_state.history,
                    }
                )

                if response.status_code == 200:
                    data    = response.json()
                    answer  = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.markdown(
                                    f"**{source['filename']}** — page {source['page']} "
                                    f"*(relevance: {source['score']})*"
                                )

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                    st.session_state.history.append({
                        "question": prompt,
                        "answer":   answer,
                    })

                else:
                    error = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the backend is running.")
