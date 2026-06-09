"""Streamlit chat UI.

Shallow glue over the query pipeline: shows the generated answer alongside the retrieved
source chunks so you can see what the answer was grounded in. Run with:

    streamlit run rag/app.py

Assumes `rag ingest` has already populated the index. Multi-turn / follow-up questions are
out of scope for v1 (the core is structured to allow them later).
"""

from __future__ import annotations

import streamlit as st

from rag.config import Config
from rag.factory import load_pipeline


@st.cache_resource
def _pipeline():
    return load_pipeline(Config.from_env())


st.set_page_config(page_title="Local RAG — ChatPDF", page_icon="📄")
st.title("📄 Local RAG over your documents")

question = st.chat_input("Ask a question about your documents")
if question:
    st.chat_message("user").write(question)
    answer = _pipeline().ask(question)

    with st.chat_message("assistant"):
        st.write(answer.text)
        st.divider()
        st.caption("Sources")
        for i, src in enumerate(answer.sources, start=1):
            with st.expander(f"[{i}] {src.document} — page {src.page}"):
                st.write(src.text)
