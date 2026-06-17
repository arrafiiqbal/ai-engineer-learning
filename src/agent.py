import os
import sqlite3
import pandas as pd
import chromadb
from typing import Annotated, TypedDict
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ── Setup ────────────────────────────────────────────────────
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

print("✅ Agent components initialized\n")


# ── Tool 1: RAG Search ──────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search the AI engineering knowledge base for concepts, 
    definitions, and explanations (RAG, embeddings, ChromaDB, etc).
    Use this for conceptual or definitional questions."""

    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    docs = results["documents"][0]
    return "\n\n".join(docs)


# ── Tool 2: SQL Database Query ───────────────────────────────
@tool
def query_sales_database(question: str) -> str:
    """Query the sales database to answer questions about 
    revenue, customers, products, regions, or sales trends.
    Use this for any business data or numbers question."""

    schema = """
Table: products | Columns: product_id, product_name, category, unit_price
Table: customers | Columns: customer_id, customer_name, region, segment
Table: sales | Columns: sale_id, product_id, customer_id, quantity, revenue, sale_date
"""

    sql_prompt = f"""
<role>You are an expert SQLite analyst.</role>

<database_schema>
{schema}
</database_schema>

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

    # Clean markdown if present
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


# ── Agent State ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Agent Node — the "Thought" step ──────────────────────────
def agent_node(state: AgentState):
    """LLM decides what to do next: call a tool or give final answer."""
    system_msg = SystemMessage(content="""You are a senior data analyst AI agent.
You have access to two tools:
1. search_knowledge_base - for AI/technical concept questions
2. query_sales_database - for business data and revenue questions

Use tools when needed. You can use multiple tools if the question requires it.
Once you have enough information, give a clear, concise final answer.
Always synthesize tool results into a coherent business-friendly response.""")

    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── Tool Node — the "Action" step ────────────────────────────
def tool_node(state: AgentState):
    """Execute whichever tool(s) the agent requested."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  🔧 Calling tool: {tool_name}({tool_args})")

        if tool_name == "search_knowledge_base":
            result = search_knowledge_base.invoke(tool_args)
        elif tool_name == "query_sales_database":
            result = query_sales_database.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}


# ── Routing Logic — decide what happens next ─────────────────
def should_continue(state: AgentState):
    """Check if the LLM wants to use a tool or is done."""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "continue"
    return "end"


# ── Build the Graph ──────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")  # loop back after tool runs

agent_graph = workflow.compile()

print("✅ Agent graph compiled\n")


# ── Run the Agent ────────────────────────────────────────────
def ask_agent(question: str):
    print(f"\n{'='*60}")
    print(f"🔍 Question: {question}")
    print(f"{'='*60}")

    result = agent_graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    final_answer = result["messages"][-1].content
    print(f"\n✅ Final Answer:\n{final_answer}")
    return final_answer


# ── Test It ──────────────────────────────────────────────────
if __name__ == "__main__":

    ask_agent("What is RAG and how does it relate to hallucination?")

    ask_agent("What is our total revenue by product category?")

    ask_agent("Which customer generates the most revenue, and can you explain what embeddings are?")
