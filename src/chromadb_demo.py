import chromadb
from sentence_transformers import SentenceTransformer

# ── Setup ────────────────────────────────────────────────────
# Local embedding model (same one from before)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB client - stores data in memory for now
client = chromadb.Client()

# A "collection" = a table in SQL terms
# This is where your documents live
collection = client.create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

print("✅ ChromaDB client created")
print("✅ Collection 'ai_knowledge_base' created\n")


# ── Step 1: Index Documents ──────────────────────────────────
print("=" * 45)
print("📌 STEP 1: Indexing Documents")
print("=" * 45)

documents = [
    "RAG stands for Retrieval Augmented Generation. It combines a retriever and a generator to produce accurate answers.",
    "Vector databases store embeddings and allow fast similarity search across millions of documents.",
    "LangChain is a framework for building applications powered by large language models.",
    "Embeddings convert text into numerical vectors that capture semantic meaning.",
    "ChromaDB is an open source vector database designed for AI applications.",
    "Prompt engineering is the process of designing inputs to get better outputs from LLMs.",
    "Python is the most popular language for AI and data science development.",
    "SQL is used to query relational databases and is essential for data analysts.",
    "Cosine similarity measures the angle between two vectors to determine how similar they are.",
    "Fine-tuning trains a pre-existing model on new data to specialize its behavior.",
]

# Generate embeddings for all documents
embeddings = embedding_model.encode(documents).tolist()

# Add to ChromaDB
# ids = unique identifier for each document (required)
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✅ Indexed {len(documents)} documents into ChromaDB\n")


# ── Step 2: Basic Semantic Search ───────────────────────────
print("=" * 45)
print("📌 STEP 2: Basic Semantic Search")
print("=" * 45)

def search(query: str, top_k: int = 3):
    """Search ChromaDB for relevant documents."""
    
    # Embed the query
    query_embedding = embedding_model.encode(query).tolist()
    
    # Query ChromaDB - it handles all the similarity math
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"🔍 Query: '{query}'")
    print(f"Top {top_k} results:\n")
    
    for i, (doc, distance) in enumerate(zip(
        results["documents"][0],
        results["distances"][0]
    ), 1):
        # ChromaDB returns distance (lower = more similar)
        # Convert to similarity score (higher = more similar)
        similarity = 1 - distance
        bar = "█" * int(similarity * 20)
        print(f"  {i}. Similarity: {similarity:.3f} {bar}")
        print(f"     {doc}\n")

search("What is retrieval augmented generation?")
search("How do I store vectors efficiently?")
search("What should I learn as a data analyst moving to AI?")


# ── Step 3: Add Metadata ─────────────────────────────────────
print("=" * 45)
print("📌 STEP 3: Search With Metadata Filtering")
print("=" * 45)

# Create a new collection with metadata
collection_meta = client.create_collection(
    name="ai_knowledge_meta",
    metadata={"hnsw:space": "cosine"}
)

documents_with_meta = [
    {
        "text": "RAG stands for Retrieval Augmented Generation.",
        "category": "concept",
        "week": 4
    },
    {
        "text": "ChromaDB is an open source vector database.",
        "category": "tool",
        "week": 3
    },
    {
        "text": "LangChain is a framework for building LLM applications.",
        "category": "tool",
        "week": 4
    },
    {
        "text": "Prompt engineering improves LLM output quality.",
        "category": "concept",
        "week": 2
    },
    {
        "text": "Embeddings convert text into numerical vectors.",
        "category": "concept",
        "week": 3
    },
]

texts = [d["text"] for d in documents_with_meta]
metadatas = [{"category": d["category"], "week": d["week"]} for d in documents_with_meta]
embeddings_meta = embedding_model.encode(texts).tolist()

collection_meta.add(
    documents=texts,
    embeddings=embeddings_meta,
    metadatas=metadatas,
    ids=[f"meta_doc_{i}" for i in range(len(texts))]
)

# Search only within "tool" category
query = "What tools should I use for AI engineering?"
query_embedding = embedding_model.encode(query).tolist()

results = collection_meta.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"category": "tool"}  # metadata filter
)

print(f"🔍 Query: '{query}'")
print(f"Filter: category = 'tool'\n")
for i, doc in enumerate(results["documents"][0], 1):
    print(f"  {i}. {doc}")


# ── Step 4: Persistent Storage ───────────────────────────────
print("\n" + "=" * 45)
print("📌 STEP 4: Persistent Storage")
print("=" * 45)

# Save to disk so data survives between sessions
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load existing collection
persistent_collection = persistent_client.get_or_create_collection(
    name="persistent_knowledge",
    metadata={"hnsw:space": "cosine"}
)

persistent_collection.add(
    documents=["This data persists between sessions!"],
    embeddings=embedding_model.encode(
        ["This data persists between sessions!"]
    ).tolist(),
    ids=["persist_test"]
)

count = persistent_collection.count()
print(f"✅ Persistent collection has {count} document(s)")
print(f"✅ Data saved to ./chroma_db folder")
print(f"\n🎉 ChromaDB demo complete!")
print(f"Next step: Connect this to an LLM → that's RAG!")