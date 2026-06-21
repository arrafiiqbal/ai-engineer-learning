# 🤖 AI Engineering Learning Journey
> Senior Data Analyst exploring AI Engineering | Self-Study Sprint | by Iqbal Arrafiiqbal

![Progress](https://img.shields.io/badge/Status-Complete-10b981)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20LangGraph%20%7C%20FastAPI%20%7C%20Docker-6366f1)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit%20Cloud-ff4b4b)
![Backend](https://img.shields.io/badge/Backend-Railway-10b981)
![Model](https://img.shields.io/badge/LLM-Llama%203.1%20via%20Groq-f59e0b)

### 🔗 Live Demo

- **💬 Chat UI:** [ai-engineer-learning…streamlit.app](https://ai-engineer-learning-73uyrb4ksxcttwsdma6dzm.streamlit.app) — friendly chat interface, just type and ask
- **⚙️ API docs:** [ai-engineer-learning-production…/docs](https://ai-engineer-learning-production.up.railway.app/docs) — interactive Swagger UI for the raw API

## 🏗️ Architecture

The frontend and backend are deployed independently as separate services that communicate over HTTP — mirroring how production systems are typically structured.

```
┌──────────────────────┐         ┌───────────────────────────┐
│  Streamlit Chat UI   │  HTTP   │   FastAPI Backend          │
│  (Streamlit Cloud)   │ ──────▶ │   (Railway)                │
│                      │  POST   │                            │
│  • streamlit         │  /ask   │   • LangGraph agent        │
│  • requests          │ ◀────── │   • ChromaDB (RAG)         │
│  only — no ML deps   │  JSON   │   • SQLite (SQL tool)      │
└──────────────────────┘         │   • Groq LLM               │
                                  └───────────────────────────┘
```

**Why decoupled?**
- **Independent scaling** — UI and agent scale separately
- **Lightweight frontend** — installs only `streamlit` + `requests`, deploys in seconds
- **Heavy backend isolated** — the PyTorch/ML stack stays containerized on Railway
- **Realistic structure** — how frontend/backend are split in real production systems

---

## 📋 About This Repo

A self-study sprint into AI Engineering, built from a Senior Data Analyst background — done purely for learning, not as an active job search. Originally scoped as an 8-week, 1–2 hr/day plan, but completed in **roughly 4 focused working sessions** since the material moved faster than planned. Documenting it here mostly as a personal reference and a way to retain what was learned.

**Actual time to complete:** ~4 working sessions (originally scoped for 8 weeks)
**Device:** MacBook Air M1 + GitHub Codespaces
**Primary API:** Groq (Llama 3.1 8B) — free, fast, no credit card required
**Deployed on:** Railway
**Tutor:** Claude (Anthropic) as AI engineering study partner

---

## 🗺️ Topics Covered

| Module | Topic | Phase | Status |
|--------|-------|-------|--------|
| 1 | Python & API Quickstart | Foundation | ✅ Complete |
| 2 | Prompt Engineering | Foundation | ✅ Complete |
| 3 | Embeddings & Vector Search | RAG Core | ✅ Complete |
| 4 | RAG App | RAG Core | ✅ Complete |
| 5 | LangChain Chains & Tools | Agents | ✅ Complete |
| 6 | AI Agent (LangGraph) | Agents | ✅ Complete |
| 7 | FastAPI + Docker Deployment | Deploy | ✅ Complete |
| 8 | Documentation & Wrap-up | Deploy | ✅ Complete |

---

## 🚀 Try It Yourself

**Live API:** [ai-engineer-learning-production.up.railway.app/docs](https://ai-engineer-learning-production.up.railway.app/docs)

1. Click **POST /ask**
2. Click **"Try it out"**
3. Enter a question:
   ```json
   {"question": "What is RAG?"}
   ```
4. Click **"Execute"**

**Try asking it:**
- `"What is RAG?"` → pulls from the AI knowledge base
- `"What's our total revenue by category?"` → queries the live SQL database
- `"Who's our top customer and what are embeddings?"` → autonomously uses both tools

---

## 📚 What I've Learned

### ✅ Module 1 — Python & API Quickstart
First LLM API call via Groq. Built a CLI chatbot with conversation history. `asyncio.gather()` made 3 parallel LLM calls run in 2s instead of 6s.

**Key insight:** LLM memory is an illusion — you manage the full conversation history yourself.

```
src/test_groq.py · src/async_demo.py · src/async_groq.py · src/chatbot.py
```

---

### ✅ Module 2 — Prompt Engineering
Compared zero-shot vs few-shot vs chain-of-thought. Built 5 reusable prompt templates with XML tags. Watched the chatbot confidently hallucinate "RAG = Red Amber Green."

**Key insight:** Hallucination is the model guessing confidently — no prompt trick fixes missing knowledge.

```
src/week2_prompting.py · src/prompt_templates.py · src/test_templates.py
```

---

### ✅ Module 3 — Embeddings & Vector Search
Text → 384 numbers, similar meaning = similar numbers. Built cosine similarity from scratch, then ChromaDB for production vector search with metadata filtering.

**Key insight:** Semantic search finds relevant content even when zero words match.

```
src/embeddings_demo.py · src/chromadb_demo.py
```

---

### ✅ Module 4 — RAG App
Connected retrieval + generation into a full RAG pipeline. Fixed the Module 2 hallucination permanently. Shipped a live Streamlit chatbot with a sources panel.

**Key insight:** The model isn't smarter with RAG — it just has the answer in front of it instead of guessing.

```
src/rag_pipeline.py · src/app.py
```

---

### ✅ Module 5 — LangChain Chains & Tools
LCEL (`prompt | llm | parser`) — pipe chaining like pandas. Sequential chains, routing chains, tool use. Natural-language-to-SQL pipeline through 3 rounds of prompt iteration.

**Key insight:** LLM generates SQL, Python executes it safely — never let the LLM touch the database directly.

```
src/week5_chains.py · src/sql_chain.py
```

---

### ✅ Module 6 — AI Agent
Built an autonomous LangGraph agent using the ReAct pattern. The agent wraps the RAG tool and SQL tool and **decides on its own** which to call. Added reasoning transparency UI + LangSmith tracing.

**Key insight:** Agents loop, chains don't. The agent decided which tools to use — nobody wrote the if/else logic.

```
src/agent.py · src/agent_app.py
```

---

### ✅ Module 7 — FastAPI + Docker Deployment

Wrapped the Module 6 agent in a production FastAPI service and deployed it live on Railway. This week involved real debugging — not just following steps.

**Bugs found and fixed (via LangSmith trace analysis):**

| # | Problem | Root Cause | Fix |
|---|---------|------------|-----|
| 1 | Agent looped 5-6 times calling the same tool | Small model (8B) doesn't reliably recognize "I have enough info" | Code-enforced tool-call counter that forces a stop, never trusting the LLM's own judgment |
| 2 | Forced stop produced an empty answer | LLM cut off mid-reasoning before generating text | Added `force_answer_node` — a separate LLM call with no tool access, forced to answer |
| 3 | Forced answer sometimes said "I don't have enough info" despite having the data | Grounding instructions were too conservative about messy/extra columns | Relaxed instruction to use partial/relevant data instead of refusing outright |
| 4 | Model described tools instead of answering ("the first tool returned...") | No instruction to synthesize naturally | Added explicit "answer as if you know this directly, never mention tools" rule |

**Key insight:** Never trust the LLM alone to terminate a loop — always add a deterministic, code-enforced safety net. And apply the same RAG-style grounding pattern to every fallback path, not just the main pipeline.

**Infrastructure lesson:** Local Docker builds repeatedly failed with `no space left on device` — not a Dockerfile bug, but a genuine constraint of building a PyTorch + ML stack inside a 32GB Codespace overlay filesystem. The fix wasn't more optimization — it was recognizing the build should happen on Railway's infrastructure instead.

**Deployment result:** Live public REST API with auto-generated `/docs` (Swagger UI) — verified working by testing directly from iPad Safari with zero terminal access needed.

```
src/main.py · Dockerfile · .dockerignore
```

---

### ✅ Module 8 — Documentation & Wrap-up

Final step — consolidating everything into a clear, well-documented record for future reference.

- Added a live demo link to this README
- Documented the full debugging journey as a technical narrative
- Reviewed the whole sprint end to end

**Key insight:** Writing things down clearly, even just for yourself, makes the knowledge stick — and makes it much easier to pick back up months from now.

---

## 🛠️ Projects Built

### 1. RAG Knowledge Assistant
A chat interface that answers questions grounded in a custom knowledge base.

**Features:** Semantic search (ChromaDB) · Sources panel per answer · Honest refusal for out-of-scope questions

**Stack:** ChromaDB · Groq API · Streamlit · SentenceTransformers · Python

---

### 2. AI Data Analyst Agent
An autonomous agent that decides for itself whether to query data, search a knowledge base, or both.

**Features:** LangGraph ReAct loop · Multi-tool autonomous selection · Reasoning transparency panel · LangSmith tracing

**Stack:** LangGraph · LangChain · Groq API · SQLite · Streamlit · LangSmith

---

### 3. Deployed Agent API
**[Try it now →](https://ai-engineer-learning-production.up.railway.app/docs)**

The agent above, hardened against 4 real reliability bugs and deployed as a public REST API.

**Features:** FastAPI endpoints with auto-generated docs · Dockerized · Deployed on Railway · Grounded fallback logic preventing hallucination

**Stack:** FastAPI · Docker · Railway · LangGraph · Groq API

---

## 🧠 Key Insights Across All Modules

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
├── src/                        # Backend — runs on Railway
│   ├── test_groq.py           # Module 1: First LLM API call
│   ├── async_demo.py          # Module 1: sync vs async comparison
│   ├── async_groq.py          # Module 1: parallel LLM calls
│   ├── chatbot.py             # Module 1: CLI chatbot with memory
│   ├── week2_prompting.py     # Module 2: prompting techniques
│   ├── prompt_templates.py    # Module 2: reusable template library
│   ├── test_templates.py      # Module 2: template tests
│   ├── embeddings_demo.py     # Module 3: embeddings + similarity
│   ├── chromadb_demo.py       # Module 3: vector database
│   ├── rag_pipeline.py        # Module 4: RAG pipeline
│   ├── app.py                 # Module 4: Streamlit RAG app (local)
│   ├── week5_chains.py        # Module 5: LangChain LCEL chains
│   ├── sql_chain.py           # Module 5: NL to SQL pipeline
│   ├── agent.py                # Module 6: LangGraph agent (CLI)
│   ├── agent_app.py            # Module 6: Streamlit agent UI (local)
│   └── main.py                 # Module 7: FastAPI production service
├── frontend/                   # Frontend — runs on Streamlit Cloud
│   ├── streamlit_ui.py        # Lightweight chat UI (calls Railway API)
│   ├── requirements.txt       # Just streamlit + requests
│   └── README.md              # Frontend architecture notes
├── chroma_db/                  # Persisted vector database
├── sales.db                    # SQLite sales database
├── Dockerfile                  # Container definition (backend)
├── .dockerignore                # Excludes unnecessary files from image
├── requirements.txt            # Backend dependencies (heavy ML stack)
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

**Or just try the live deployed version:** [ai-engineer-learning-production.up.railway.app/docs](https://ai-engineer-learning-production.up.railway.app/docs) — no setup required.

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

---

## 🎓 What's Next

This sprint is complete — finished in about 4 working sessions rather than the originally planned 8 weeks. Possible directions to keep exploring, no fixed timeline:
- Multi-agent orchestration (CrewAI)
- Production observability dashboards
- Expanding the knowledge base with real-world documents
- Cost optimization for LLM API usage at scale

---

*A self-study record, kept mostly for personal reference.*
