import os
import sqlite3
import pandas as pd
import chromadb
from typing import Annotated, TypedDict, Literal
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── Two models: smart for the supervisor, fast for specialists ──
llm_supervisor = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile"  # routing needs good reasoning
)
llm_worker = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant"     # specialists do simpler, focused jobs
)

# ── Shared resources ─────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="rag_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# ── Shared State ─────────────────────────────────────────────
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str          # the original user question
    data_findings: str     # what the researcher found
    concept_findings: str  # what the explainer found
    next_agent: str        # supervisor's routing decision
    needs_data: bool        # NEW
    needs_concept: bool     # NEW
    answer_written: bool    # NEW


# ── Specialist 1: Researcher ─────────────────────────────────
def researcher_node(state: TeamState):
    """Queries the sales database."""
    question = state["question"]

    schema = """
Table: products | Columns: product_id, product_name, category, unit_price
Table: customers | Columns: customer_id, customer_name, region, segment
Table: sales | Columns: sale_id, product_id, customer_id, quantity, revenue, sale_date
"""
    sql_prompt = f"""You are an expert SQLite analyst. Schema:
{schema}
Rules: JOIN when data spans tables (category is in products, region is in customers).
Use STRFTIME() for dates. Return ONLY raw SQL, no markdown.
Question: {question}"""

    sql_query = llm_worker.invoke(sql_prompt).content.strip()
    if "```" in sql_query:
        sql_query = "\n".join(
            l for l in sql_query.split("\n") if not l.startswith("```")
        ).strip()

    try:
        conn = sqlite3.connect("./sales.db")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        findings = df.to_string(index=False) if not df.empty else "No data found."
    except Exception as e:
        findings = f"SQL Error: {e}"

    return {
        "data_findings": findings,
        "messages": [AIMessage(content=f"[Researcher] Found data:\n{findings}")]
    }


# ── Specialist 2: Explainer ──────────────────────────────────
def explainer_node(state: TeamState):
    """Searches the knowledge base for AI/ML concepts."""
    question = state["question"]

    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Using ONLY this context, explain the AI/ML concept in the question.
Context: {context}
Question: {question}
Give a clear, concise explanation."""

    explanation = llm_worker.invoke(prompt).content

    return {
        "concept_findings": explanation,
        "messages": [AIMessage(content=f"[Explainer] {explanation}")]
    }


# ── Specialist 3: Writer ─────────────────────────────────────
def writer_node(state: TeamState):
    """Combines findings into a final answer."""
    question = state["question"]
    data = state.get("data_findings", "")
    concept = state.get("concept_findings", "")

    prompt = f"""Write a clear, business-friendly answer to the question below.
Use the findings provided. Do not mention tools, agents, or how you got the info.

Question: {question}

Data findings: {data if data else "(none)"}

Concept findings: {concept if concept else "(none)"}

Final answer:"""

    final = llm_worker.invoke(prompt).content

    return {
        "messages": [AIMessage(content=final)],
        "answer_written": True
    }


# ── Supervisor ───────────────────────────────────────────────
def supervisor_node(state: TeamState):
    """Rule-based routing — deterministic, no infinite loops."""

    # On first visit, decide what the question NEEDS (one LLM call, once)
    if "needs_data" not in state or state.get("needs_data") is None:
        classify_prompt = f"""Analyze this question and answer with two yes/no flags.

Question: "{state['question']}"

Does it require BUSINESS DATA (revenue, customers, products, sales)? 
Does it require an AI/ML CONCEPT explanation (RAG, embeddings, etc.)?

Reply in EXACTLY this format:
DATA: yes or no
CONCEPT: yes or no"""

        response = llm_supervisor.invoke(classify_prompt).content.lower()
        needs_data = "data: yes" in response
        needs_concept = "concept: yes" in response
    else:
        needs_data = state["needs_data"]
        needs_concept = state["needs_concept"]

    # Deterministic routing based on what's needed vs what's done
    has_data = bool(state.get("data_findings"))
    has_concept = bool(state.get("concept_findings"))
    has_answer = state.get("answer_written", False)

    if needs_data and not has_data:
        next_agent = "researcher"
    elif needs_concept and not has_concept:
        next_agent = "explainer"
    elif not has_answer:
        next_agent = "writer"
    else:
        next_agent = "FINISH"

    print(f"  🧭 Supervisor routes to: {next_agent}  (needs_data={needs_data}, needs_concept={needs_concept})")

    return {
        "next_agent": next_agent,
        "needs_data": needs_data,
        "needs_concept": needs_concept
    }



# ── Routing function (reads supervisor's decision) ───────────
def route_decision(state: TeamState) -> Literal["researcher", "explainer", "writer", "__end__"]:
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return END
    return next_agent


# ── Build the Graph ──────────────────────────────────────────
workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("explainer", explainer_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", route_decision)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("explainer", "supervisor")
workflow.add_edge("writer", "supervisor")

team_graph = workflow.compile()


# ── Run It ───────────────────────────────────────────────────
def run_team(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")

    result = team_graph.invoke(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "data_findings": "",
            "concept_findings": "",
            "next_agent": "",
            "needs_data": None,
            "needs_concept": None,
            "answer_written": False
        },
        config={"recursion_limit": 15}
    )

    final_answer = result["messages"][-1].content
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(final_answer)
    return final_answer


if __name__ == "__main__":
    run_team("What is RAG?")
