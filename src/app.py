import os
import sys
import streamlit as st
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔍",
    layout="wide"
)

# ── Initialize (cached so it only runs once) ─────────────────
@st.cache_resource
def init_pipeline():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    return embedding_model, groq_client, collection

embedding_model, groq_client, collection = init_pipeline()

# ── RAG Functions ────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0], results["distances"][0]

def generate(query: str, context_docs: list[str]) -> str:
    context = "\n\n".join([
        f"Document {i+1}: {doc}"
        for i, doc in enumerate(context_docs)
    ])

    prompt = f"""
<context>
{context}
</context>

<instructions>
    Answer the question using ONLY the information in the context above.
    If the answer is not in the context, say "I don't have information about that."
    Never make up information.
    Be concise and direct.
</instructions>

<question>{query}</question>
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a precise AI assistant. Answer only from the provided context. Never hallucinate."
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ── UI ───────────────────────────────────────────────────────
st.title("🔍 RAG Knowledge Assistant")
st.caption("Answers grounded in your knowledge base — no hallucinations")

# Sidebar — knowledge base stats
with st.sidebar:
    st.header("📚 Knowledge Base")
    doc_count = collection.count()
    st.metric("Documents indexed", doc_count)

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Your question is embedded into a vector")
    st.markdown("2. ChromaDB finds the most relevant documents")
    st.markdown("3. Groq generates an answer from those documents only")

    st.divider()
    top_k = st.slider("Documents to retrieve", 1, 5, 3)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources used"):
                for i, (source, dist) in enumerate(
                    zip(message["sources"], message["distances"]), 1
                ):
                    similarity = 1 - dist
                    st.markdown(f"**Source {i}** — Similarity: `{similarity:.3f}`")
                    st.caption(source)

# Chat input
if query := st.chat_input("Ask anything about AI engineering..."):

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Generate RAG response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            docs, distances = retrieve(query, top_k=top_k)

        with st.spinner("🤖 Generating answer..."):
            answer = generate(query, docs)

        st.markdown(answer)

        # Show sources in expander
        with st.expander("📄 Sources used"):
            for i, (source, dist) in enumerate(
                zip(docs, distances), 1
            ):
                similarity = 1 - dist
                st.markdown(f"**Source {i}** — Similarity: `{similarity:.3f}`")
                st.caption(source)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
        "distances": distances
    })
