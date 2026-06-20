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
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage


from fastapi import FastAPI
from pydantic import BaseModel



# ── LangSmith Tracing ────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-data-analyst-agent"

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


# ── Tools (same as Week 6) ───────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search ONLY for AI/ML conceptual questions: definitions, explanations,
    how things work. Examples: 'What is RAG?', 'What are embeddings?'.
    Do NOT use this for business data, revenue, or sales questions."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    docs = results["documents"][0]
    return "\n\n".join(docs)


@tool
def query_sales_database(question: str) -> str:
    """Query ONLY for business data questions: revenue, customers, products,
    regions, sales trends. Examples: 'Total revenue by category?', 'Top customer?'.
    Do NOT use this for AI/ML concept questions."""

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


# ── Agent Graph (same as Week 6) ─────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    system_msg = SystemMessage(content="""You are a senior data analyst AI agent with two tools.

TOOL 1: search_knowledge_base
- Use ONLY for questions about AI/ML concepts, definitions, or explanations
- Examples: "What is RAG?", "What are embeddings?", "How does ChromaDB work?"

TOOL 2: query_sales_database
- Use ONLY for questions about revenue, customers, products, regions, or sales numbers
- Examples: "What's our top revenue category?", "Who is our best customer?"

STRICT RULES:
1. Read the question carefully. Does it ask about an AI/ML CONCEPT, or about BUSINESS DATA/NUMBERS?
2. Call ONLY the tool that matches. Most questions need exactly ONE tool.
3. Only call BOTH tools if the question explicitly asks about both a concept AND business data in the same sentence.
4. NEVER connect or mix information between the two tools unless the question explicitly asks for both.
5. Call each tool AT MOST ONCE.
6. As soon as you have a relevant tool result, answer immediately. Do not call more tools.
7. NEVER invent a relationship between sales data and AI concepts unless the user explicitly asks about that relationship.""")

    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

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

def force_answer_node(state: AgentState):
    """Force a final answer using ONLY the actual retrieved tool results,
    not the model's general knowledge."""
    
    # Extract ONLY the tool result messages — nothing else
    tool_results = [
        msg.content for msg in state["messages"]
        if isinstance(msg, ToolMessage)
    ]
    
    original_question = state["messages"][0].content
    
    if not tool_results:
        return {"messages": [AIMessage(content="I don't have enough retrieved information to answer this question.")]}
    
    context = "\n\n".join(tool_results)
    
    prompt = f"""
<context>
{context}
</context>

<instructions>
    Answer the question using ONLY the information in the context above.
    If the context does not contain a clear answer, say "I don't have enough information to answer this."
    Do NOT use any outside knowledge.
</instructions>

<question>{original_question}</question>
"""
    
    response = llm.invoke(prompt)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    tool_call_count = sum(
        1 for msg in state["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
    )

    if tool_call_count >= 2:
        return "force_answer"

    if last_message.tool_calls:
        return "continue"
    return "end"



workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("force_answer", force_answer_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "force_answer": "force_answer",
        "end": END
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("force_answer", END)

agent_graph = workflow.compile()

# Safety: cap total steps to prevent infinite/excessive looping
MAX_ITERATIONS = 6



# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="AI Data Analyst Agent API",
    description="An autonomous agent that queries sales data and AI knowledge",
    version="1.0.0"
)


# ── Request/Response Models ──────────────────────────────────
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str
    tools_used: list[str]


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "AI Data Analyst Agent",
        "endpoints": ["/ask", "/health"]
    }


@app.get("/health")
def health_check():
    """Verify the service and its dependencies are working."""
    return {
        "status": "healthy",
        "knowledge_base_docs": collection.count()
    }


@app.post("/ask", response_model=Answer)
def ask_agent(payload: Question):
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=payload.question)]},
        config={"recursion_limit": MAX_ITERATIONS}
    )

    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

    final_answer = result["messages"][-1].content

    return Answer(
        question=payload.question,
        answer=final_answer,
        tools_used=tools_used
    )


