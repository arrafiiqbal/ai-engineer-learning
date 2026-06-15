import numpy as np
from sentence_transformers import SentenceTransformer

# Load a free, local embedding model
# Downloads once (~90MB), runs on your machine
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Demo 1: See what an embedding looks like ────────────────
print("=" * 45)
print("📌 DEMO 1: What does an embedding look like?")
print("=" * 45)

text = "I love working with data"
embedding = model.encode(text)

print(f"Text: '{text}'")
print(f"Embedding shape: {embedding.shape}")  # (384,) = 384 numbers
print(f"First 5 values: {embedding[:5].round(4)}")


# ── Demo 2: Semantic similarity ──────────────────────────────
print("\n" + "=" * 45)
print("📌 DEMO 2: Semantic Similarity")
print("=" * 45)

def cosine_similarity(a, b):
    """Measure similarity between two embeddings. 
    Returns 0 (different) to 1 (identical)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentences = [
    "I love working with data",        # anchor
    "I enjoy analyzing datasets",      # similar meaning
    "Data analysis is my passion",     # similar meaning  
    "The weather is nice today",       # unrelated
    "Python is great for data science" # somewhat related
]

anchor = model.encode(sentences[0])
others = model.encode(sentences[1:])

print(f"Anchor: '{sentences[0]}'\n")
for i, (sentence, embedding) in enumerate(zip(sentences[1:], others)):
    similarity = cosine_similarity(anchor, embedding)
    bar = "█" * int(similarity * 20)
    print(f"Similarity: {similarity:.3f} {bar}")
    print(f"Text: '{sentence}'\n")


# ── Demo 3: Semantic Search ──────────────────────────────────
print("=" * 45)
print("📌 DEMO 3: Semantic Search")
print("=" * 45)

# Simulate a small knowledge base (like documents in your DA work)
knowledge_base = [
    "RAG stands for Retrieval Augmented Generation",
    "Vector databases store embeddings for fast search",
    "Python is a popular programming language for data science",
    "LangChain is a framework for building LLM applications",
    "Embeddings convert text into numerical vectors",
    "Cosine similarity measures the angle between two vectors",
    "SQL is used to query relational databases",
    "Machine learning models learn patterns from data",
]

# Embed the entire knowledge base
print("Indexing knowledge base...")
kb_embeddings = model.encode(knowledge_base)
print(f"Indexed {len(knowledge_base)} documents\n")

def semantic_search(query: str, top_k: int = 3):
    """Find most relevant documents for a query."""
    query_embedding = model.encode(query)
    
    # Calculate similarity to every document
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in kb_embeddings
    ]
    
    # Sort by similarity, get top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"🔍 Query: '{query}'")
    print(f"Top {top_k} results:\n")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Score: {similarities[idx]:.3f}")
        print(f"     {knowledge_base[idx]}\n")

# Test with 3 different queries
semantic_search("What is retrieval augmented generation?")
semantic_search("How do I store vectors?")
semantic_search("How do databases work?")