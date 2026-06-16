# 🤖 AI Engineering Learning Journey

> Senior Data Analyst → AI Engineer | 8-Week Sprint | by Iqbal Arrafiiqbal

![Progress](https://img.shields.io/badge/Progress-5%2F8%20Weeks-a855f7)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20LangChain%20%7C%20ChromaDB%20%7C%20Groq-6366f1)
![Environment](https://img.shields.io/badge/Environment-GitHub%20Codespaces-10b981)
![Model](https://img.shields.io/badge/LLM-Llama%203.1%20via%20Groq-f59e0b)

-----

## 📋 About This Repo

This repository documents my structured self-study journey from **Senior Data Analyst to AI Engineer** — built in public, one week at a time.

**Daily commitment:** 1–2 hours/day  
**Device:** MacBook Air M1 + GitHub Codespaces  
**Primary API:** Groq (Llama 3.1 8B) — free, fast, no credit card required  
**Tutor:** Claude (Anthropic) as AI engineering expert

-----

## 🗺️ 8-Week Roadmap

|Week|Topic                      |Phase     |Status    |
|----|---------------------------|----------|----------|
|1   |Python & API Quickstart    |Foundation|✅ Complete|
|2   |Prompt Engineering         |Foundation|✅ Complete|
|3   |Embeddings & Vector Search |RAG Core  |✅ Complete|
|4   |RAG App                    |RAG Core  |✅ Complete|
|5   |LangChain Chains & Tools   |Agents    |✅ Complete|
|6   |Build an AI Agent          |Agents    |🔜 Next    |
|7   |FastAPI + Docker Deployment|Deploy    |⏳ Upcoming|
|8   |Portfolio Polish & Launch  |Deploy    |⏳ Upcoming|

-----

## 📚 What I’ve Learned

### ✅ Week 1 — Python & API Quickstart

**Goal:** Get dev environment ready and make first LLM API call.

**Key concepts:**

- GitHub Codespaces as cloud dev environment
- API keys via GitHub Secrets — never hardcoded in code
- `async/await` with `asyncio.gather()` — 3 parallel LLM calls in 2s instead of 6s
- LLM conversation memory — stateless API, history managed manually as a list

**Key insight:** LLM memory is an illusion you create by sending the full conversation history on every API call.

**Files:**

```
src/test_groq.py       → First LLM API call via Groq
src/async_demo.py      → sync vs async speed comparison
src/async_groq.py      → 3 parallel LLM calls with asyncio.gather
src/chatbot.py         → CLI chatbot with conversation history
```

-----

### ✅ Week 2 — Prompt Engineering

**Goal:** Master structured prompting for reliable, production-quality LLM outputs.

**Key concepts:**

- Zero-shot vs few-shot vs chain-of-thought — same model, different results
- XML tags for structured output — `<task>`, `<context>`, `<instructions>`, `<input>`
- System prompts — set persona before conversation starts
- 5 reusable prompt templates using Python dataclasses

**Key insight:** Prompt engineering changes output quality without changing the model. XML tags give format control; system prompts give persona control.

**Real example:**

```
Zero-shot  → "RAG = Red Amber Green" ❌ (hallucination)
Few-shot   → NEUTRAL ✅ (learned from examples)
Chain-of-thought → NEUTRAL ✅ (reasoned through it)
```

**Files:**

```
src/week2_prompting.py    → Zero-shot, few-shot, CoT, XML, system prompts
src/prompt_templates.py   → 5 reusable prompt templates (dataclass)
src/test_templates.py     → Test all 5 templates in parallel
```

-----

### ✅ Week 3 — Embeddings & Vector Search

**Goal:** Understand how semantic search works — the backbone of every RAG system.

**Key concepts:**

- Embeddings: text → 384 numbers where similar meaning = similar numbers
- Cosine similarity: built from scratch with numpy (same as correlation in stats)
- ChromaDB: production vector database — collection = SQL table, query() = SELECT ORDER BY similarity
- Metadata filtering: SQL WHERE clause combined with semantic search

**Key insight:** Semantic search finds relevant content even when zero words match — it operates on meaning, not keywords.

**Real example:**

```python
# These sentences are 0.678 similar — different words, same meaning
"I love working with data"
"I enjoy analyzing datasets"

# This is only 0.124 similar — completely unrelated
"The weather is nice today"
```

**Files:**

```
src/embeddings_demo.py    → Embeddings, cosine similarity, semantic search
src/chromadb_demo.py      → Vector DB: index, search, metadata filter, persist
```

-----

### ✅ Week 4 — RAG App (Portfolio Project #1)

**Goal:** Fix hallucination permanently. Build and deploy a live RAG chatbot.

**Key concepts:**

- RAG pipeline: index_documents() → retrieve() → generate()
- Schema injection: ground LLM with real DB structure before asking it to query
- Grounded prompting: `<context>` XML tag + “answer ONLY from context” instruction
- Streamlit UI with sources panel — full transparency per answer

**Key insight:** The model isn’t smarter with RAG — it just has the answer in front of it. It reads instead of guesses.

**Before vs After RAG:**

```
Without RAG → "RAG = Red Amber Green" ❌
With RAG    → "RAG = Retrieval Augmented Generation" ✅

Out of scope: "What is the capital of France?"
Without RAG → "Paris" (hallucinated confidently)
With RAG    → "I don't have information about that." ✅
```

**Files:**

```
src/rag_pipeline.py    → Full RAG pipeline: index → retrieve → generate
src/app.py             → Streamlit RAG chatbot UI (Portfolio Project #1)
```

-----

### ✅ Week 5 — LangChain Chains & Tools

**Goal:** Go beyond single prompts — build multi-step chains and give LLM tools.

**Key concepts:**

- LCEL (LangChain Expression Language): `prompt | llm | parser` — pipe operator chains components
- Sequential chains: output of Chain 1 feeds into Chain 2
- Routing chains: classify question first → route to different chain
- Tool use: `@tool` decorator + `llm.bind_tools()` — LLM autonomously decides which tool to call
- SQL chain: natural language → SQL → execute → business insight (3 rounds of prompt iteration)

**Key insight:** LLM generates SQL, Python executes it safely. Never let the LLM touch the database directly.

**Prompt iteration in practice:**

```
Round 1 → 2/4 queries working  (missing JOIN rules)
Round 2 → 3/4 queries working  (wrong SQL dialect: EXTRACT vs STRFTIME)
Round 3 → 4/4 queries working  ✅
```

**Files:**

```
src/week5_chains.py    → LCEL: simple, sequential, routing, tool use chains
src/sql_chain.py       → NL → SQL → execute → business insight pipeline
```

-----

## 🚀 Portfolio Projects

### Project 1: RAG Knowledge Assistant ✅ Live

A chat interface that answers questions grounded in a custom knowledge base.

**Features:**

- Semantic search over indexed documents (ChromaDB)
- Sources panel per answer — full transparency
- Honest refusal for out-of-scope questions
- Adjustable retrieval depth (1–5 documents)

**Stack:** ChromaDB · Groq API · Streamlit · SentenceTransformers · Python

-----

### Project 2: Data Analyst Agent 🔜 Week 6

An autonomous agent that takes a business question and decides for itself what to do.

**Planned features:**

- Queries SQLite database autonomously
- Searches RAG knowledge base when needed
- Synthesizes multi-source insights
- Returns grounded, actionable business recommendations

**Stack:** LangGraph · LangChain · Groq API · SQLite · FastAPI · Streamlit

-----

## 🧠 Key Insights Across All Weeks

|# |Insight                                                                   |
|--|--------------------------------------------------------------------------|
|1 |LLM memory is an illusion — you manage conversation history yourself      |
|2 |`asyncio.gather()` makes parallel LLM calls 3x faster than sequential     |
|3 |Prompt engineering changes output quality without changing the model      |
|4 |XML tags = format control. System prompts = persona control               |
|5 |Hallucination = model guessing confidently. RAG is the only real cure     |
|6 |Embeddings = GPS coordinates for meaning in 384-dimensional space         |
|7 |ChromaDB collection ≈ SQL table. `query()` ≈ `SELECT ORDER BY similarity` |
|8 |The model isn’t smarter with RAG — it just has the answer in front of it  |
|9 |Honest refusal beats confident wrong answer every time                    |
|10|LCEL pipes are to LangChain what method chaining is to pandas             |
|11|LLM generates SQL, Python executes it. Never let LLM touch DB directly    |
|12|Production prompts are never right first try. Observe → add rules → repeat|

-----

## 🛠️ Tech Stack

|Category       |Tools                                  |
|---------------|---------------------------------------|
|Language       |Python 3.12                            |
|LLM API        |Groq (Llama 3.1 8B Instant)            |
|LLM Framework  |LangChain, LCEL                        |
|Vector DB      |ChromaDB                               |
|Embeddings     |SentenceTransformers (all-MiniLM-L6-v2)|
|UI             |Streamlit                              |
|Database       |SQLite                                 |
|Environment    |GitHub Codespaces                      |
|Version Control|Git + GitHub                           |

-----

## 📁 Repository Structure

```
ai-engineer-learning/
├── .devcontainer/
│   └── devcontainer.json      # Auto-installs packages on Codespace start
├── src/
│   ├── test_groq.py           # Week 1: First LLM API call
│   ├── async_demo.py          # Week 1: sync vs async comparison
│   ├── async_groq.py          # Week 1: parallel LLM calls
│   ├── chatbot.py             # Week 1: CLI chatbot with memory
│   ├── week2_prompting.py     # Week 2: prompting techniques
│   ├── prompt_templates.py    # Week 2: reusable template library
│   ├── test_templates.py      # Week 2: template tests
│   ├── embeddings_demo.py     # Week 3: embeddings + similarity
│   ├── chromadb_demo.py       # Week 3: vector database
│   ├── rag_pipeline.py        # Week 4: RAG pipeline
│   ├── app.py                 # Week 4: Streamlit RAG app
│   ├── week5_chains.py        # Week 5: LangChain LCEL chains
│   └── sql_chain.py           # Week 5: NL to SQL pipeline
├── chroma_db/                 # Persisted vector database
├── sales.db                   # SQLite sales database
├── requirements.txt           # All dependencies pinned
└── README.md                  # This file
```

-----

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/arrafiiqbal/ai-engineer-learning
cd ai-engineer-learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (GitHub Codespaces secret or .env)
export GROQ_API_KEY=your-key-here

# 4. Run the RAG app
streamlit run src/app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

-----

*Updated after every learning session. Follow along as I build toward AI Engineering. 🚀*