import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-data-analyst-agent"

import sqlite3
import pandas as pd
import streamlit as st
import chromadb
from typing import Annotated, TypedDict
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🕵️",
    layout="wide"
)

# ── Initialize (cached so it only runs once) ─────────────────
@st.cache_resource
def init_agent():
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.1-8b-instant"
    )
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    return llm, embedding_model, collection

llm, embedding_model, collection = init_agent()


# ── Tools ────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search the AI engineering knowledge base for concepts,
    definitions, and explanations (RAG, embeddings, ChromaDB, etc).
    Use this for conceptual or definitional questions."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    docs = results["documents"][0]
    return "\n\n".join(docs)


@tool
def query_sales_database(question: str) -> str:
    """Query the sales database to answer questions about
    revenue, customers, products, regions, or sales trends."""

    schema = """
Table: products | Columns: product_id, product_name, category, unit_price
Table: customers | Columns: customer_id, customer_name, region, segment
Table: sales | Columns: sale_id, product_id, customer_id, quantity, revenue, sale_date
"""

    sql_prompt = f"""
<role>You are an expert SQLite analyst.</role>
<database_schema>{schema}</database_schema>
<rules>
    - Always use JOIN when data spans multiple tables
    - The sales table has NO category or region columns
    - To get category: JOIN sales with products ON product_id
    - To get region: JOIN sales with customers ON customer_id
    - Never reference aliases with table prefix in ORDER BY
    - ALWAYS use STRFTIME() for dates, never EXTRACT()
    - Return ONLY raw SQL — no markdown, no explanation
</rules>
<question>{question}</question>
"""
    sql_query = llm.invoke(sql_prompt).content.strip()

    if "```" in sql_query:
        lines = sql_query.split("\n")
        sql_query = "\n".join([l for l in lines if not l.startswith("```")]).strip()

    try:
        conn = sqlite3.connect("./sales.db")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        if df.empty:
            return "Query returned no results."
        return f"SQL used: {sql_query}\n\nResults:\n{df.to_string(index=False)}"
    except Exception as e:
        return f"SQL Error: {e}"


tools = [search_knowledge_base, query_sales_database]
llm_with_tools = llm.bind_tools(tools)


# ── Agent Graph ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    system_msg = SystemMessage(content="""You are a senior data analyst AI agent.
You have access to two tools:
1. search_knowledge_base - for AI/technical concept questions
2. query_sales_database - for business data and revenue questions

Use tools when needed. You can use multiple tools if the question requires it.
Once you have enough information, give a clear, concise final answer.""")
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_results = []
    tool_log = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "search_knowledge_base":
            result = search_knowledge_base.invoke(tool_args)
        elif tool_name == "query_sales_database":
            result = query_sales_database.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        tool_log.append({"tool": tool_name, "args": tool_args, "result": result})
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    # Stash the log in session state so the UI can show it
    if "tool_log" not in st.session_state:
        st.session_state.tool_log = []
    st.session_state.tool_log.extend(tool_log)

    return {"messages": tool_results}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

@st.cache_resource
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()

agent_graph = build_graph()


# ── UI ───────────────────────────────────────────────────────
st.title("🕵️ AI Data Analyst Agent")
st.caption("Autonomously decides whether to query data, search knowledge, or both")

with st.sidebar:
    st.header("🛠️ Available Tools")
    st.markdown("**📊 query_sales_database**")
    st.caption("Revenue, customers, products, regions")
    st.markdown("**📚 search_knowledge_base**")
    st.caption("AI/ML concepts and definitions")

    st.divider()
    st.markdown("**Try asking:**")
    st.caption("• What's our top revenue category?")
    st.caption("• What is RAG?")
    st.caption("• Top customer AND what are embeddings?")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_log" not in st.session_state:
    st.session_state.tool_log = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about sales data or AI concepts..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Clear tool log for this run
    st.session_state.tool_log = []

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is reasoning..."):
            result = agent_graph.invoke({
                "messages": [HumanMessage(content=query)]
            })
            final_answer = result["messages"][-1].content

        st.markdown(final_answer)

        # Show the agent's reasoning steps
        if st.session_state.tool_log:
            with st.expander(f"🔍 Agent used {len(st.session_state.tool_log)} tool call(s) — click to see reasoning"):
                for i, log in enumerate(st.session_state.tool_log, 1):
                    st.markdown(f"**Step {i}: `{log['tool']}`**")
                    st.caption(f"Input: {log['args']}")
                    st.text(log['result'][:500])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

