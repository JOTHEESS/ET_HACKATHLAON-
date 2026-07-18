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
import streamlit.components.v1 as components

from ingest.graph_builder import load_or_build_graph
from ingest.path_viz import build_path_html
from ingest.synthesis import synthesize
from ingest.vector_builder import load_or_build_vector_store

st.set_page_config(page_title="ET Industrial Knowledge Intelligence", page_icon="🛠️")

CONFIDENCE_BADGE = {"high": "🟢", "medium": "🟡", "low": "🔴"}


@st.cache_resource
def load_resources():
    # First run on a fresh clone: chroma_db/ is gitignored, so this builds
    # the vector store from data/corpus (local embedding only, no API key).
    with st.spinner("Loading indexes (a fresh clone builds them from the corpus, ~a minute)..."):
        return load_or_build_vector_store(), load_or_build_graph()


def render_details(graph, matched_entities: list, retrieved_doc_ids: list,
                   citations: list = None) -> None:
    with st.expander("Retrieval details"):
        st.write(f"**Matched entities:** {', '.join(matched_entities) or 'none'}")
        st.write(f"**Retrieved documents:** {', '.join(retrieved_doc_ids) or 'none'}")
    with st.expander("Why these documents — graph paths"):
        html = build_path_html(graph, matched_entities, citations or retrieved_doc_ids)
        if html:
            components.html(html, height=440)
            st.caption("🟢 query entities · 🔵 linking entities · 🟧 retrieved documents — drag nodes, hover for aliases")
        else:
            st.caption("No graph entities matched in the query — this retrieval was vector-only.")


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
            render_details(graph, msg["details"]["matched_entities"],
                           msg["details"]["retrieved_doc_ids"],
                           msg["details"].get("citations"))

query = st.chat_input('Ask about the P-204 incident, e.g. "Was there early warning before the failure?"')

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Set ANTHROPIC_API_KEY above to run synthesis.")
        else:
            # Prior turns (excluding the query just appended) let follow-ups
            # like "was it acted on?" resolve against the earlier answer.
            history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state.messages[:-1]]
            result = None
            try:
                with st.spinner("Retrieving and synthesizing..."):
                    result = synthesize(query, collection=collection, graph=graph,
                                        history=history)
            except Exception as e:
                st.error(f"Synthesis failed ({type(e).__name__}): {e}\n\n"
                         "Check the API key and network connection, then ask again.")

            if result is not None:
                confidence = result.get("confidence", "low")
                badge = CONFIDENCE_BADGE.get(confidence, "⚪")
                if result.get("cached"):
                    badge = f"⚡ {badge}"
                citations = ", ".join(result["citations"]) or "none"
                answer_block = (
                    f"{result['answer']}\n\n"
                    f"{badge} **Confidence:** {confidence}  |  **Citations:** {citations}"
                )

                st.markdown(result["answer"])
                st.markdown(f"{badge} **Confidence:** {confidence}  |  **Citations:** {citations}")
                render_details(graph, result["matched_entities"],
                               result["retrieved_doc_ids"], result["citations"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_block,
                    "details": {
                        "matched_entities": result["matched_entities"],
                        "retrieved_doc_ids": result["retrieved_doc_ids"],
                        "citations": result["citations"],
                    },
                })
