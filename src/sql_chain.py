import os
import sqlite3
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Setup ────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant"
)
parser = StrOutputParser()


# ── Step 1: Create Sample Database ──────────────────────────
def create_sample_db():
    """Create a realistic sales database for demo."""
    conn = sqlite3.connect("./sales.db")
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            unit_price REAL
        );

        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT,
            region TEXT,
            segment TEXT
        );

        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            product_id INTEGER,
            customer_id INTEGER,
            quantity INTEGER,
            revenue REAL,
            sale_date TEXT,
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    """)

    # Insert sample data
    products = [
        (1, "Analytics Pro", "Software", 299.99),
        (2, "Data Viz Tool", "Software", 199.99),
        (3, "ML Platform", "Software", 499.99),
        (4, "API Gateway", "Infrastructure", 149.99),
        (5, "Cloud Storage", "Infrastructure", 99.99),
    ]

    customers = [
        (1, "Acme Corp", "North", "Enterprise"),
        (2, "TechStart", "South", "SMB"),
        (3, "DataCo", "East", "Enterprise"),
        (4, "CloudBase", "West", "SMB"),
        (5, "AI Ventures", "North", "Enterprise"),
        (6, "QuickData", "South", "SMB"),
    ]

    sales = [
        (1,  1, 1, 5,  1499.95, "2026-01-15"),
        (2,  2, 2, 3,   599.97, "2026-01-22"),
        (3,  3, 3, 2,   999.98, "2026-02-05"),
        (4,  1, 4, 1,   299.99, "2026-02-14"),
        (5,  4, 5, 8,  1199.92, "2026-02-28"),
        (6,  5, 6, 10,  999.90, "2026-03-10"),
        (7,  3, 1, 3,  1499.97, "2026-03-15"),
        (8,  2, 3, 5,   999.95, "2026-03-22"),
        (9,  1, 5, 4,  1199.96, "2026-04-01"),
        (10, 4, 2, 6,   899.94, "2026-04-15"),
        (11, 5, 4, 15, 1499.85, "2026-04-20"),
        (12, 3, 6, 1,   499.99, "2026-05-01"),
        (13, 2, 1, 7,  1399.93, "2026-05-10"),
        (14, 1, 3, 2,   599.98, "2026-05-18"),
        (15, 4, 5, 4,   599.96, "2026-06-01"),
    ]

    cursor.executemany(
        "INSERT INTO products VALUES (?,?,?,?)", products)
    cursor.executemany(
        "INSERT INTO customers VALUES (?,?,?,?)", customers)
    cursor.executemany(
        "INSERT INTO sales VALUES (?,?,?,?,?,?)", sales)

    conn.commit()
    conn.close()
    print("✅ Sales database created (sales.db)\n")


# ── Step 2: Get DB Schema ─────────────────────────────────────
def get_schema() -> str:
    """Extract schema from database to inject into prompt."""
    conn = sqlite3.connect("./sales.db")
    cursor = conn.cursor()

    schema = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_defs = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
        schema.append(f"Table: {table_name} | Columns: {col_defs}")

    conn.close()
    return "\n".join(schema)


# ── Step 3: Execute SQL ───────────────────────────────────────
def execute_sql(query: str) -> str:
    """Run SQL query and return results as string."""
    try:
        conn = sqlite3.connect("./sales.db")

        # Clean up LLM output — strip markdown if present
        clean_query = query.strip()
        if "```" in clean_query:
            lines = clean_query.split("\n")
            clean_query = "\n".join([
                l for l in lines
                if not l.startswith("```")
            ]).strip()

        df = pd.read_sql_query(clean_query, conn)
        conn.close()

        if df.empty:
            return "Query returned no results."

        return df.to_string(index=False)

    except Exception as e:
        return f"SQL Error: {e}"


# ── Step 4: Build the SQL Chain ──────────────────────────────
def build_sql_chain():
    schema = get_schema()

    # Step 1 — natural language → SQL
    sql_prompt = ChatPromptTemplate.from_template("""
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
    - Use plain alias names in ORDER BY (e.g. ORDER BY avg_revenue)
    - ALWAYS use STRFTIME() for dates, never EXTRACT()
    - For year: STRFTIME('%Y', date_column)
    - For month: STRFTIME('%m', date_column)
    - Return ONLY raw SQL — no markdown, no backticks, no explanation
</rules>

<question>{question}</question>
""")

    # Step 2 — SQL results → plain English insight
    insight_prompt = ChatPromptTemplate.from_template("""
<role>You are a senior data analyst presenting to business stakeholders.</role>

<question>{question}</question>

<sql_query>{sql_query}</sql_query>

<results>{results}</results>

<instructions>
    Write a concise business insight (2-3 sentences).
    Use plain English, no technical jargon.
    Focus on what this means for the business.
    Start with the most important number or finding.
</instructions>
""")

    sql_generation_chain = sql_prompt | llm | parser

    def full_sql_chain(question: str) -> dict:
        print(f"\n🔍 Question: {question}")

        # Generate SQL
        sql_query = sql_generation_chain.invoke({
            "schema": schema,
            "question": question
        })
        print(f"📝 SQL Generated:\n   {sql_query.strip()}")

        # Execute SQL
        results = execute_sql(sql_query)
        print(f"📊 Results:\n{results}")

        # Generate insight
        insight = (insight_prompt | llm | parser).invoke({
            "question": question,
            "sql_query": sql_query,
            "results": results
        })

        return {
            "question": question,
            "sql": sql_query.strip(),
            "results": results,
            "insight": insight
        }

    return full_sql_chain


# ── Run Tests ────────────────────────────────────────────────
if __name__ == "__main__":
    create_sample_db()

    schema = get_schema()
    print("📋 Database Schema:")
    print(schema)
    print()

    sql_chain = build_sql_chain()

    questions = [
        "What is the total revenue by product category?",
        "Which customer has generated the most revenue?",
        "What are the monthly revenue trends in 2026?",
        "Which region has the highest average deal size?",
    ]

    print("=" * 50)
    for question in questions:
        result = sql_chain(question)
        print(f"\n💡 Insight:\n{result['insight']}")
        print("-" * 50)
