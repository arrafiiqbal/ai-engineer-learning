import os
import asyncio
from groq import AsyncGroq
from prompt_templates import ai_tutor_template

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

conversation_history = []

async def chat(user_message: str, week: int = 2) -> str:
    
    # Build system prompt from template
    system = ai_tutor_template.system
    
    # First message gets full tutor context injected
    if len(conversation_history) == 0:
        first_message = ai_tutor_template.render(
            week=week,
            concept=user_message
        )
        conversation_history.append({
            "role": "user",
            "content": first_message
        })
    else:
        # Subsequent messages go in naturally
        conversation_history.append({
            "role": "user",
            "content": user_message
        })

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            *conversation_history
        ]
    )

    assistant_message = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message


async def main():
    print("=" * 45)
    print("🤖 AI Engineering Tutor — Week 2 Upgrade")
    print("=" * 45)
    print("Your personal AI tutor, context-aware")
    print("Commands: 'quit' to exit | 'clear' to reset")
    print("=" * 45)

    week = 2

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Great session! Don't forget to commit your work.")
            break

        if not user_input:
            continue

        # Clear history and start fresh
        if user_input.lower() == "clear":
            conversation_history.clear()
            print("🔄 Conversation reset.")
            continue

        print("🤔 Thinking...", end="\r")

        try:
            response = await chat(user_input, week=week)
            print(" " * 20, end="\r")
            print(f"\n🤖 Tutor: {response}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


asyncio.run(main())