import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Setup ────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant"
)

parser = StrOutputParser()

print("✅ LangChain + Groq initialized\n")


# ── Chain 1: Simple Prompt Chain ─────────────────────────────
print("=" * 50)
print("📌 CHAIN 1: Simple Prompt Chain")
print("=" * 50)

prompt1 = ChatPromptTemplate.from_template("""
<role>You are a concise AI engineering tutor.</role>
<task>Explain {concept} in exactly 2 sentences.</task>
""")

# Build the chain using pipe operator
chain1 = prompt1 | llm | parser

result1 = chain1.invoke({"concept": "vector embeddings"})
print(f"Result: {result1}\n")


# ── Chain 2: Sequential Chain (output feeds into next) ───────
print("=" * 50)
print("📌 CHAIN 2: Sequential Chain")
print("=" * 50)

# Step 1 — explain the concept
explain_prompt = ChatPromptTemplate.from_template("""
<task>Explain {concept} in 2 sentences for a data analyst.</task>
""")

# Step 2 — generate a quiz question from the explanation
quiz_prompt = ChatPromptTemplate.from_template("""
<context>{explanation}</context>
<task>
    Generate one multiple choice quiz question to test 
    understanding of the above explanation.
    Format: Question + 4 options (A/B/C/D) + correct answer.
</task>
""")

# Chain them: explain → quiz
explain_chain = explain_prompt | llm | parser
quiz_chain = quiz_prompt | llm | parser

# RunnablePassthrough passes original input alongside
from langchain_core.runnables import RunnablePassthrough

sequential_chain = (
    {"explanation": explain_chain, "concept": RunnablePassthrough()}
    | quiz_prompt
    | llm
    | parser
)

result2 = sequential_chain.invoke({"concept": "RAG (Retrieval Augmented Generation)"})
print(f"Result:\n{result2}\n")


# ── Chain 3: Routing Chain ───────────────────────────────────
print("=" * 50)
print("📌 CHAIN 3: Routing Chain")
print("=" * 50)

# First classify the question type
classify_prompt = ChatPromptTemplate.from_template("""
<task>
    Classify this question into exactly one category.
    Return ONLY the category name, nothing else.
    
    Categories:
    - TECHNICAL (about code, tools, frameworks)
    - CONCEPTUAL (about ideas, theory, definitions)
    - CAREER (about jobs, learning, transition)
</task>

<question>{question}</question>
""")

# Different prompts for different routes
technical_prompt = ChatPromptTemplate.from_template("""
<role>You are a senior AI engineer.</role>
<task>Answer this technical question with a code example: {question}</task>
""")

conceptual_prompt = ChatPromptTemplate.from_template("""
<role>You are an AI educator.</role>  
<task>Explain this concept clearly with an analogy: {question}</task>
""")

career_prompt = ChatPromptTemplate.from_template("""
<role>You are a career coach for AI engineers.</role>
<task>Give practical career advice for: {question}</task>
""")

def route(info):
    """Route to the right chain based on question type."""
    question = info["question"]
    category = (classify_prompt | llm | parser).invoke({"question": question}).strip()
    print(f"  → Classified as: {category}")

    if "TECHNICAL" in category:
        return technical_prompt | llm | parser
    elif "CAREER" in category:
        return career_prompt | llm | parser
    else:
        return conceptual_prompt | llm | parser

from langchain_core.runnables import RunnableLambda

routing_chain = RunnableLambda(route)

questions = [
    {"question": "How do I install ChromaDB?"},
    {"question": "What is cosine similarity?"},
    {"question": "Should I learn LangChain or LlamaIndex first?"},
]

for q in questions:
    print(f"\n🔍 Question: {q['question']}")
    result = routing_chain.invoke(q)
    print(f"✅ Answer: {result[:200]}...")


# ── Chain 4: Tool Use ────────────────────────────────────────
print("\n" + "=" * 50)
print("📌 CHAIN 4: Tool Use")
print("=" * 50)

from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_week_topic(week: int) -> str:
    """Get the learning topic for a given week of the AI engineering sprint."""
    topics = {
        1: "Python & API Quickstart",
        2: "Prompt Engineering",
        3: "Embeddings & Vector Search",
        4: "RAG App",
        5: "LangChain Chains & Tools",
        6: "AI Agent",
        7: "FastAPI + Docker Deployment",
        8: "Portfolio Polish & Launch"
    }
    return topics.get(week, "Unknown week")

tools = [calculate, get_week_topic]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

tool_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{question}")
])

tool_questions = [
    "What is 1234 multiplied by 5678?",
    "What topic am I studying in week 5?",
]

for question in tool_questions:
    print(f"\n🔍 Question: {question}")
    response = (tool_prompt | llm_with_tools).invoke({"question": question})

    # Check if the model wants to use a tool
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  🔧 Using tool: {tool_name}({tool_args})")

            # Execute the right tool
            if tool_name == "calculate":
                result = calculate.invoke(tool_args)
            elif tool_name == "get_week_topic":
                result = get_week_topic.invoke(tool_args)

            print(f"  ✅ Tool result: {result}")
    else:
        print(f"  ✅ Direct answer: {response.content}")
