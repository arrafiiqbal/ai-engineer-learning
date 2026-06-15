import os
import asyncio
from groq import AsyncGroq

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

async def ask(question: str) -> str:
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

async def main():
    # Ask 3 questions simultaneously
    results = await asyncio.gather(
        ask("What is machine learning in one sentence?"),
        ask("What is an API in one sentence?"),
        ask("What is Python in one sentence?"),
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nQ{i}: {result}")

asyncio.run(main())