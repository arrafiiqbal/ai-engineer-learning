import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Setup ────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="rag_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

print("✅ RAG pipeline initialized\n")


# ── Step 1: Index Documents ──────────────────────────────────
def index_documents(documents: list[dict]):
    """
    Add documents to ChromaDB.
    Each document: {"id": str, "text": str, "metadata": dict}
    """
    texts = [d["text"] for d in documents]
    ids = [d["id"] for d in documents]
    metadatas = [d.get("metadata", {}) for d in documents]
    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Indexed {len(documents)} documents\n")


# ── Step 2: Retrieve Relevant Documents ─────────────────────
def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Find most relevant documents for a query."""
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# ── Step 3: Generate Grounded Answer ────────────────────────
def generate(query: str, context_docs: list[str]) -> str:
    """Generate answer using retrieved context."""

    # Format context documents
    context = "\n\n".join([
        f"Document {i+1}: {doc}"
        for i, doc in enumerate(context_docs)
    ])

    # RAG prompt template
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
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ── Step 4: Full RAG Pipeline ────────────────────────────────
def rag(query: str) -> dict:
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents
    2. Generate grounded answer
    """
    print(f"🔍 Query: {query}")

    # Retrieve
    docs = retrieve(query, top_k=3)
    print(f"📄 Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"   {i}. {doc[:80]}...")

    # Generate
    print("\n🤖 Generating answer...")
    answer = generate(query, docs)

    return {
        "query": query,
        "sources": docs,
        "answer": answer
    }


# ── Test It ──────────────────────────────────────────────────
if __name__ == "__main__":

    # Index AI engineering knowledge base
    documents = [
        {
            "id": "rag_001",
            "text": "RAG stands for Retrieval Augmented Generation. It is a technique that combines a retrieval system with a language model. Instead of relying on the model's training data, RAG retrieves relevant documents and injects them into the prompt as context before generating an answer.",
            "metadata": {"topic": "RAG", "week": 4}
        },
        {
            "id": "rag_002",
            "text": "The RAG pipeline has three main steps: indexing, retrieval, and generation. Indexing converts documents into embeddings stored in a vector database. Retrieval finds the most relevant documents for a query using semantic search. Generation uses an LLM to produce an answer grounded in the retrieved documents.",
            "metadata": {"topic": "RAG", "week": 4}
        },
        {
            "id": "embed_001",
            "text": "Embeddings are numerical representations of text. They convert words and sentences into vectors of numbers where similar meanings are placed close together in vector space. The model all-MiniLM-L6-v2 produces embeddings with 384 dimensions.",
            "metadata": {"topic": "embeddings", "week": 3}
        },
        {
            "id": "chroma_001",
            "text": "ChromaDB is an open source vector database. It stores embeddings alongside the original documents and metadata. You can query it with a vector to find the most similar documents using cosine similarity or L2 distance.",
            "metadata": {"topic": "ChromaDB", "week": 3}
        },
        {
            "id": "prompt_001",
            "text": "Prompt engineering is the practice of designing inputs to language models to get better outputs. Techniques include zero-shot prompting, few-shot prompting with examples, chain-of-thought reasoning, and XML tags for structured output.",
            "metadata": {"topic": "prompting", "week": 2}
        },
        {
            "id": "hallucination_001",
            "text": "Hallucination in LLMs refers to when a model generates confident but factually incorrect information. RAG reduces hallucination by grounding the model's response in retrieved documents. The model is instructed to answer only from the provided context.",
            "metadata": {"topic": "hallucination", "week": 4}
        },
        {
            "id": "groq_001",
            "text": "Groq is an AI inference platform that provides free API access to open source models including Llama 3.1. It is significantly faster than other providers and requires no credit card for the free tier.",
            "metadata": {"topic": "tools", "week": 1}
        },
        {
            "id": "async_001",
            "text": "Async programming in Python uses async and await keywords. asyncio.gather() runs multiple coroutines simultaneously. For LLM applications, async allows multiple API calls to run in parallel instead of sequentially, reducing total wait time significantly.",
            "metadata": {"topic": "python", "week": 1}
        },
    ]

    # Only index if collection is empty
    if collection.count() == 0:
        index_documents(documents)
    else:
        print(f"✅ Collection already has {collection.count()} documents\n")

    print("=" * 50)
    print("📌 TEST 1: The hallucination test")
    print("=" * 50)
    result = rag("What is RAG?")
    print(f"\n✅ Answer:\n{result['answer']}")

    print("\n" + "=" * 50)
    print("📌 TEST 2: Out of context question")
    print("=" * 50)
    result2 = rag("What is the capital of France?")
    print(f"\n✅ Answer:\n{result2['answer']}")

    print("\n" + "=" * 50)
    print("📌 TEST 3: Multi-document retrieval")
    print("=" * 50)
    result3 = rag("How does ChromaDB work with embeddings?")
    print(f"\n✅ Answer:\n{result3['answer']}")