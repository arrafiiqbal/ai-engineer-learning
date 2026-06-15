import os
import asyncio
from groq import AsyncGroq
from prompt_templates import (
    sentiment_template,
    summarizer_template,
    code_review_template,
    data_analyst_template,
    ai_tutor_template
)

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

async def run_template(template, label: str, **kwargs) -> None:
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": template.system},
            {"role": "user", "content": template.render(**kwargs)}
        ]
    )
    print(f"\n{'='*40}")
    print(f"📌 {label}")
    print(f"{'='*40}")
    print(response.choices[0].message.content)

async def main():
    await asyncio.gather(

        run_template(
            sentiment_template,
            "SENTIMENT TEMPLATE",
            text="The new dashboard loads instantly but the filters are really confusing."
        ),

        run_template(
            summarizer_template,
            "SUMMARIZER TEMPLATE",
            text="""Machine learning is a subset of artificial intelligence 
            that gives systems the ability to automatically learn and improve 
            from experience without being explicitly programmed. It focuses on 
            developing computer programs that can access data and use it to 
            learn for themselves.""",
            max_words=20
        ),

        run_template(
            code_review_template,
            "CODE REVIEW TEMPLATE",
            code="""
def get_user(users, id):
    for u in users:
        if u['id'] == id:
            return u
    return None
"""
        ),

        run_template(
            data_analyst_template,
            "DATA ANALYST TEMPLATE",
            context="Monthly sales report for Q2 2026",
            data="Revenue: $420K (+12% MoM), Churn: 8% (-2% MoM), New users: 1,240 (+34% MoM)"
        ),

        run_template(
            ai_tutor_template,
            "AI TUTOR TEMPLATE",
            week=2,
            concept="What is a vector embedding and why does it matter for RAG?"
        ),

    )

asyncio.run(main())