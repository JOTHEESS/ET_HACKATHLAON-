"""
Streamlit chat UI for the Hybrid GraphRAG pipeline.

Ask a question about the P-204 pump maintenance corpus and get an
answer synthesized from the knowledge graph + vector store hybrid
retriever, with citations and a confidence rating.

Usage:
    streamlit run app.py
"""
import os

import streamlit as st

from ingest.graph_builder import load_graph
from ingest.synthesis import synthesize
from ingest.vector_builder import load_vector_store

st.set_page_config(page_title="ET Industrial Knowledge Intelligence", page_icon="🛠️")

CONFIDENCE_BADGE = {"high": "🟢", "medium": "🟡", "low": "🔴"}


@st.cache_resource
def load_resources():
    return load_vector_store(), load_graph()


def render_details(matched_entities: list, retrieved_doc_ids: list) -> None:
    with st.expander("Retrieval details"):
        st.write(f"**Matched entities:** {', '.join(matched_entities) or 'none'}")
        st.write(f"**Retrieved documents:** {', '.join(retrieved_doc_ids) or 'none'}")


st.title("Industrial Knowledge Intelligence")
st.caption("Hybrid GraphRAG over the P-204 pump maintenance corpus")

if not os.environ.get("ANTHROPIC_API_KEY"):
    api_key = st.text_input(
        "ANTHROPIC_API_KEY",
        type="password",
        help="Not found in the environment - paste your key here to run synthesis.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

if "messages" not in st.session_state:
    st.session_state.messages = []

collection, graph = load_resources()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "details" in msg:
            render_details(msg["details"]["matched_entities"], msg["details"]["retrieved_doc_ids"])

query = st.chat_input('Ask about the P-204 incident, e.g. "Was there early warning before the failure?"')

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Set ANTHROPIC_API_KEY above to run synthesis.")
        else:
            with st.spinner("Retrieving and synthesizing..."):
                result = synthesize(query, collection=collection, graph=graph)

            confidence = result.get("confidence", "low")
            badge = CONFIDENCE_BADGE.get(confidence, "⚪")
            citations = ", ".join(result["citations"]) or "none"
            answer_block = (
                f"{result['answer']}\n\n"
                f"{badge} **Confidence:** {confidence}  |  **Citations:** {citations}"
            )

            st.markdown(result["answer"])
            st.markdown(f"{badge} **Confidence:** {confidence}  |  **Citations:** {citations}")
            render_details(result["matched_entities"], result["retrieved_doc_ids"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_block,
                "details": {
                    "matched_entities": result["matched_entities"],
                    "retrieved_doc_ids": result["retrieved_doc_ids"],
                },
            })
