import os
import asyncio
from groq import AsyncGroq

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

# This list stores the full conversation history
# Groq/LLMs need the entire history every call to "remember"
conversation_history = []

SYSTEM_PROMPT = """You are a helpful AI engineering tutor. 
You help students learn AI engineering concepts clearly and concisely.
Keep responses focused and practical."""

async def chat(user_message: str) -> str:
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history  # unpack full history
        ]
    )
    
    assistant_message = response.choices[0].message.content
    
    # Add assistant response to history so it remembers
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

async def main():
    print("🤖 AI Engineering Tutor")
    print("=" * 40)
    print("Type your question or 'quit' to exit")
    print("=" * 40)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Exit condition
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 See you next session!")
            break
            
        # Skip empty input
        if not user_input:
            continue
        
        # Show thinking indicator
        print("🤔 Thinking...", end="\r")
        
        try:
            response = await chat(user_input)
            # Clear thinking indicator
            print(" " * 20, end="\r")
            print(f"\n🤖 Tutor: {response}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

asyncio.run(main())