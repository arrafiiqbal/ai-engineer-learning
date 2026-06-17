# 🤖 AI Engineering Learning Journey

> Senior Data Analyst → AI Engineer | 8-Week Sprint | by Iqbal Arrafiiqbal

![Progress](https://img.shields.io/badge/Progress-6%2F8%20Weeks-f59e0b)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20LangGraph%20%7C%20ChromaDB%20%7C%20Groq-6366f1)
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
|6   |AI Agent (LangGraph)       |Agents    |✅ Complete|
|7   |FastAPI + Docker Deployment|Deploy    |🔜 Next    |
|8   |Portfolio Polish & Launch  |Deploy    |⏳ Upcoming|

-----

## 📚 What I’ve Learned

### ✅ Week 1 — Python & API Quickstart

First LLM API call via Groq (free, no credit card). Built a CLI chatbot with conversation history. `asyncio.gather()` made 3 parallel LLM calls run in 2s instead of 6s.

**Key insight:** LLM memory is an illusion you create by sending the full conversation history on every API call.

```
src/test_groq.py · src/async_demo.py · src/async_groq.py · src/chatbot.py
```

-----

### ✅ Week 2 — Prompt Engineering

Compared zero-shot vs few-shot vs chain-of-thought. Built 5 reusable prompt templates with XML tags using Python dataclasses. Witnessed hallucination firsthand: chatbot confidently said “RAG = Red Amber Green.”

**Key insight:** XML tags = format control. System prompts = persona control. Hallucination can only be fixed by RAG, not better prompting.

```
src/week2_prompting.py · src/prompt_templates.py · src/test_templates.py
```

-----

### ✅ Week 3 — Embeddings & Vector Search

Learned embeddings: text → 384 numbers where similar meaning = similar numbers. Built cosine similarity from scratch with numpy, then replaced it with ChromaDB — a production vector database with metadata filtering.

**Key insight:** Semantic search finds relevant content even when zero words match — it operates on meaning, not keywords.

```
src/embeddings_demo.py · src/chromadb_demo.py
```

-----

### ✅ Week 4 — RAG App (Portfolio Project #1)

Connected retrieval + generation into a full RAG pipeline. Fixed the Week 2 hallucination permanently — “RAG = Retrieval Augmented Generation” with correct grounding. Shipped a live Streamlit chatbot with a sources panel.

**Key insight:** The model isn’t smarter with RAG — it just has the answer in front of it instead of guessing.

```
src/rag_pipeline.py · src/app.py
```

-----

### ✅ Week 5 — LangChain Chains & Tools

Learned LCEL (`prompt | llm | parser`) — pipe-based chaining, same concept as pandas method chaining. Built sequential chains, routing chains, and tool use with `@tool` + `bind_tools()`. Built a natural-language-to-SQL pipeline through 3 rounds of prompt iteration.

**Key insight:** LLM generates SQL, Python executes it safely — never let the LLM touch the database directly.

```
src/week5_chains.py · src/sql_chain.py
```

-----

### ✅ Week 6 — AI Agent (Portfolio Project #2)

Built an autonomous LangGraph agent using the **ReAct pattern** (Thought → Action → Observation, looped). The agent wraps the Week 4 RAG tool and Week 5 SQL tool, and **autonomously decides** which to call — including calling both for compound questions. Added a Streamlit UI with full reasoning transparency, plus LangSmith tracing for observability.

**Key insight:** Agents loop, chains don’t. Nodes + edges + state = a graph that can cycle until the LLM decides it has enough information.

**Real example:**

```
Q: "Who's our top customer and what are embeddings?"
→ Agent calls query_sales_database (gets: Acme Corp, $4,399.85)
→ Agent calls search_knowledge_base (gets: embeddings definition)
→ Agent synthesizes both into one coherent answer
(Nobody told it to call both tools — it decided on its own)
```

```
src/agent.py · src/agent_app.py
```

-----

## 🚀 Portfolio Projects

### Project 1: RAG Knowledge Assistant ✅ Live

A chat interface that answers questions grounded in a custom knowledge base.

**Features:** Semantic search (ChromaDB) · Sources panel per answer · Honest refusal for out-of-scope questions · Adjustable retrieval depth

**Stack:** ChromaDB · Groq API · Streamlit · SentenceTransformers · Python

-----

### Project 2: AI Data Analyst Agent ✅ Live

An autonomous agent that decides for itself whether to query data, search a knowledge base, or both.

**Features:** LangGraph ReAct loop · Multi-tool autonomous selection · Reasoning transparency panel · LangSmith tracing for full observability

**Stack:** LangGraph · LangChain · Groq API · SQLite · Streamlit · LangSmith

-----

## 🧠 Key Insights Across All Weeks

|# |Insight                                                                  |
|--|-------------------------------------------------------------------------|
|1 |LLM memory is an illusion — you manage conversation history yourself     |
|2 |`asyncio.gather()` makes parallel LLM calls 3x faster than sequential    |
|3 |Prompt engineering changes output quality without changing the model     |
|4 |XML tags = format control. System prompts = persona control              |
|5 |Hallucination = model guessing confidently. RAG is the only real cure    |
|6 |Embeddings = GPS coordinates for meaning in 384-dimensional space        |
|7 |ChromaDB collection ≈ SQL table. `query()` ≈ `SELECT ORDER BY similarity`|
|8 |The model isn’t smarter with RAG — it just has the answer in front of it |
|9 |LCEL pipes are to LangChain what method chaining is to pandas            |
|10|LLM generates SQL, Python executes it. Never let LLM touch DB directly   |
|11|Agents loop. Chains don’t. That’s the entire difference                  |
|12|The agent decided which tools to use — nobody wrote the if/else logic    |
|13|Reasoning transparency separates a toy agent from a trustworthy one      |

-----

## 🛠️ Tech Stack

|Category           |Tools                                  |
|-------------------|---------------------------------------|
|Language           |Python 3.12                            |
|LLM API            |Groq (Llama 3.1 8B Instant)            |
|LLM Framework      |LangChain, LCEL, LangGraph             |
|Agent Observability|LangSmith                              |
|Vector DB          |ChromaDB                               |
|Embeddings         |SentenceTransformers (all-MiniLM-L6-v2)|
|UI                 |Streamlit                              |
|Database           |SQLite                                 |
|Environment        |GitHub Codespaces                      |
|Version Control    |Git + GitHub                           |

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
│   ├── app.py                 # Week 4: Streamlit RAG app (Portfolio #1)
│   ├── week5_chains.py        # Week 5: LangChain LCEL chains
│   ├── sql_chain.py           # Week 5: NL to SQL pipeline
│   ├── agent.py                # Week 6: LangGraph agent (CLI)
│   └── agent_app.py            # Week 6: Streamlit agent UI (Portfolio #2)
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

# 5. Or run the AI agent
streamlit run src/agent_app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

-----

*Updated after every learning session. Follow along as I build toward AI Engineering. 🚀*