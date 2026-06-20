# 🤖 AI Engineering Learning Journey
> Senior Data Analyst → AI Engineer | 8-Week Sprint | by Iqbal Arrafiiqbal

![Progress](https://img.shields.io/badge/Progress-7%2F8%20Weeks-f59e0b)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20LangGraph%20%7C%20FastAPI%20%7C%20Docker-6366f1)
![Deployment](https://img.shields.io/badge/Deployed-Railway-10b981)
![Model](https://img.shields.io/badge/LLM-Llama%203.1%20via%20Groq-f59e0b)

---

## 📋 About This Repo

This repository documents my structured self-study journey from **Senior Data Analyst to AI Engineer** — built in public, one week at a time.

**Daily commitment:** 1–2 hours/day
**Device:** MacBook Air M1 + GitHub Codespaces
**Primary API:** Groq (Llama 3.1 8B) — free, fast, no credit card required
**Deployed on:** Railway
**Tutor:** Claude (Anthropic) as AI engineering expert

---

## 🗺️ 8-Week Roadmap

| Week | Topic | Phase | Status |
|------|-------|-------|--------|
| 1 | Python & API Quickstart | Foundation | ✅ Complete |
| 2 | Prompt Engineering | Foundation | ✅ Complete |
| 3 | Embeddings & Vector Search | RAG Core | ✅ Complete |
| 4 | RAG App | RAG Core | ✅ Complete |
| 5 | LangChain Chains & Tools | Agents | ✅ Complete |
| 6 | AI Agent (LangGraph) | Agents | ✅ Complete |
| 7 | FastAPI + Docker Deployment | Deploy | ✅ Complete |
| 8 | Portfolio Polish & Launch | Deploy | 🔜 Next |

---

## 📚 What I've Learned

### ✅ Week 1 — Python & API Quickstart
First LLM API call via Groq. Built a CLI chatbot with conversation history. `asyncio.gather()` made 3 parallel LLM calls run in 2s instead of 6s.

**Key insight:** LLM memory is an illusion — you manage the full conversation history yourself.

```
src/test_groq.py · src/async_demo.py · src/async_groq.py · src/chatbot.py
```

---

### ✅ Week 2 — Prompt Engineering
Compared zero-shot vs few-shot vs chain-of-thought. Built 5 reusable prompt templates with XML tags. Watched the chatbot confidently hallucinate "RAG = Red Amber Green."

**Key insight:** Hallucination is the model guessing confidently — no prompt trick fixes missing knowledge.

```
src/week2_prompting.py · src/prompt_templates.py · src/test_templates.py
```

---

### ✅ Week 3 — Embeddings & Vector Search
Text → 384 numbers, similar meaning = similar numbers. Built cosine similarity from scratch, then ChromaDB for production vector search with metadata filtering.

**Key insight:** Semantic search finds relevant content even when zero words match.

```
src/embeddings_demo.py · src/chromadb_demo.py
```

---

### ✅ Week 4 — RAG App (Portfolio Project #1)
Connected retrieval + generation into a full RAG pipeline. Fixed the Week 2 hallucination permanently. Shipped a live Streamlit chatbot with a sources panel.

**Key insight:** The model isn't smarter with RAG — it just has the answer in front of it instead of guessing.

```
src/rag_pipeline.py · src/app.py
```

---

### ✅ Week 5 — LangChain Chains & Tools
LCEL (`prompt | llm | parser`) — pipe chaining like pandas. Sequential chains, routing chains, tool use. Natural-language-to-SQL pipeline through 3 rounds of prompt iteration.

**Key insight:** LLM generates SQL, Python executes it safely — never let the LLM touch the database directly.

```
src/week5_chains.py · src/sql_chain.py
```

---

### ✅ Week 6 — AI Agent (Portfolio Project #2)
Built an autonomous LangGraph agent using the ReAct pattern. The agent wraps the RAG tool and SQL tool and **decides on its own** which to call. Added reasoning transparency UI + LangSmith tracing.

**Key insight:** Agents loop, chains don't. The agent decided which tools to use — nobody wrote the if/else logic.

```
src/agent.py · src/agent_app.py
```

---

### ✅ Week 7 — FastAPI + Docker Deployment

Wrapped the Week 6 agent in a production FastAPI service and deployed it live on Railway. This week involved real debugging — not just following steps.

**Bugs found and fixed (via LangSmith trace analysis):**

| # | Problem | Root Cause | Fix |
|---|---------|------------|-----|
| 1 | Agent looped 5-6 times calling the same tool | Small model (8B) doesn't reliably recognize "I have enough info" | Code-enforced tool-call counter that forces a stop, never trusting the LLM's own judgment |
| 2 | Forced stop produced an empty answer | LLM cut off mid-reasoning before generating text | Added `force_answer_node` — a separate LLM call with no tool access, forced to answer |
| 3 | Forced answer sometimes said "I don't have enough info" despite having the data | Grounding instructions were too conservative about messy/extra columns | Relaxed instruction to use partial/relevant data instead of refusing outright |
| 4 | Model described tools instead of answering ("the first tool returned...") | No instruction to synthesize naturally | Added explicit "answer as if you know this directly, never mention tools" rule |

**Key insight:** Never trust the LLM alone to terminate a loop — always add a deterministic, code-enforced safety net. And apply the same RAG-style grounding pattern to every fallback path, not just the main pipeline.

**Infrastructure lesson:** Local Docker builds repeatedly failed with `no space left on device` — not a Dockerfile bug, but a genuine constraint of building a PyTorch + ML stack inside a 32GB Codespace overlay filesystem. The fix wasn't more optimization — it was recognizing the build should happen on Railway's infrastructure instead, which has the resources designed for this.

**Deployment result:** Live public REST API at a Railway-generated domain, with auto-generated `/docs` (Swagger UI) — verified working by testing directly from iPad Safari with zero terminal access needed.

```
src/main.py · Dockerfile · .dockerignore
```

---

## 🚀 Portfolio Projects

### Project 1: RAG Knowledge Assistant ✅ Live
A chat interface that answers questions grounded in a custom knowledge base.

**Features:** Semantic search (ChromaDB) · Sources panel per answer · Honest refusal for out-of-scope questions

**Stack:** ChromaDB · Groq API · Streamlit · SentenceTransformers · Python

---

### Project 2: AI Data Analyst Agent ✅ Live
An autonomous agent that decides for itself whether to query data, search a knowledge base, or both.

**Features:** LangGraph ReAct loop · Multi-tool autonomous selection · Reasoning transparency panel · LangSmith tracing

**Stack:** LangGraph · LangChain · Groq API · SQLite · Streamlit · LangSmith

---

### Project 3: Production-Deployed Agent API ✅ Live
The agent above, hardened against 4 real reliability bugs and deployed as a public REST API.

**Features:** FastAPI endpoints with auto-generated docs · Dockerized · Deployed on Railway · Grounded fallback logic preventing hallucination · Tested across devices including mobile/tablet browsers

**Stack:** FastAPI · Docker · Railway · LangGraph · Groq API

---

## 🧠 Key Insights Across All Weeks

| # | Insight |
|---|---------|
| 1 | LLM memory is an illusion — you manage conversation history yourself |
| 2 | Hallucination = model guessing confidently. RAG is the only real cure |
| 3 | Embeddings = GPS coordinates for meaning in 384-dimensional space |
| 4 | RAG = retrieval + generation. Retrieval kills hallucination |
| 5 | Honest refusal beats confident wrong answer every time |
| 6 | LCEL pipes are to LangChain what method chaining is to pandas |
| 7 | Tool use = LLM deciding what to call. First spark of agent behavior |
| 8 | Agents loop. Chains don't. That's the entire difference |
| 9 | The agent decided which tools to use — nobody wrote the if/else logic |
| 10 | Never trust the LLM alone to terminate a loop — add a code-enforced limit |
| 11 | Apply the RAG grounding pattern everywhere the LLM gives a final answer |
| 12 | Disk space errors during Docker build can be a real infra limit, not a config bug |
| 13 | CI/CD platforms build with proper resources — don't fight your dev machine |
| 14 | A truly deployed API works from any device with a browser, even an iPad |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| LLM API | Groq (Llama 3.1 8B Instant) |
| LLM Framework | LangChain, LCEL, LangGraph |
| Agent Observability | LangSmith |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Backend API | FastAPI |
| Containerization | Docker |
| Deployment | Railway |
| UI | Streamlit |
| Database | SQLite |
| Dev Environment | GitHub Codespaces |
| Version Control | Git + GitHub |

---

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
│   ├── agent_app.py            # Week 6: Streamlit agent UI (Portfolio #2)
│   └── main.py                 # Week 7: FastAPI production service (Portfolio #3)
├── chroma_db/                  # Persisted vector database
├── sales.db                    # SQLite sales database
├── Dockerfile                  # Container definition
├── .dockerignore                # Excludes unnecessary files from image
├── requirements.txt            # All dependencies pinned
└── README.md                   # This file
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/arrafiiqbal/ai-engineer-learning
cd ai-engineer-learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export GROQ_API_KEY=your-key-here

# 4. Run locally
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 5. Or run the Streamlit UIs
streamlit run src/app.py        # RAG app
streamlit run src/agent_app.py  # Agent app
```

**Try the live deployed API:** visit `/docs` on the Railway URL for an interactive Swagger UI — no setup required, works from any browser.

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

---

*Updated after every learning session. Follow along as I build toward AI Engineering. 🚀*
