import os
import asyncio
from groq import AsyncGroq

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

async def ask(prompt: str, label: str) -> None:
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n{'='*40}")
    print(f"📌 {label}")
    print(f"{'='*40}")
    print(response.choices[0].message.content)

async def main():

    # 1. ZERO-SHOT
    zero_shot = """
    Classify this text as POSITIVE, NEGATIVE, or NEUTRAL.
    
    Text: "The model took forever to respond but the answer was decent."
    
    Classification:
    """

    # 2. FEW-SHOT
    few_shot = """
    Classify text as POSITIVE, NEGATIVE, or NEUTRAL.
    
    Examples:
    Text: "This tool is incredibly fast and accurate." → POSITIVE
    Text: "Completely useless, wasted my time." → NEGATIVE
    Text: "The update changed the interface." → NEUTRAL
    
    Now classify:
    Text: "The model took forever to respond but the answer was decent."
    
    Classification:
    """

    # 3. CHAIN-OF-THOUGHT
    chain_of_thought = """
    Classify text as POSITIVE, NEGATIVE, or NEUTRAL.
    
    Think through it step by step before giving your answer:
    1. What is the main sentiment?
    2. Are there mixed feelings?
    3. What is the final classification?
    
    Text: "The model took forever to respond but the answer was decent."
    """

    # 4. XML TAGS
    xml_prompt = """
    <task>
        Analyze the sentiment of the given text and provide a structured response.
    </task>
    
    <examples>
        <example>
            <text>This tool is incredibly fast and accurate.</text>
            <sentiment>POSITIVE</sentiment>
            <reason>Strong positive language with no negatives.</reason>
        </example>
        <example>
            <text>Completely useless, wasted my time.</text>
            <sentiment>NEGATIVE</sentiment>
            <reason>Strong negative language throughout.</reason>
        </example>
    </examples>
    
    <instructions>
        - Identify ALL sentiments present
        - Weigh positive vs negative elements
        - Return ONLY this format:
        SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
        REASON: [one sentence]
        CONFIDENCE: [HIGH/MEDIUM/LOW]
    </instructions>
    
    <input>
        The model took forever to respond but the answer was decent.
    </input>
    """

    # 5. SYSTEM PROMPT
    system_prompt_test = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a senior data analyst with 10 years experience 
                in NLP and sentiment analysis. You are precise, concise, and always 
                structure your answers clearly. You never guess — if uncertain, 
                you say so explicitly."""
            },
            {
                "role": "user",
                "content": "Analyze sentiment: 'The model took forever to respond but the answer was decent.'"
            }
        ]
    )

    # Run zero-shot, few-shot, chain-of-thought in parallel
    await asyncio.gather(
        ask(zero_shot, "ZERO-SHOT"),
        ask(few_shot, "FEW-SHOT"),
        ask(chain_of_thought, "CHAIN-OF-THOUGHT"),
        ask(xml_prompt, "XML TAGS"),
    )

    # Print system prompt result separately
    print(f"\n{'='*40}")
    print("📌 SYSTEM PROMPT")
    print(f"{'='*40}")
    print(system_prompt_test.choices[0].message.content)

asyncio.run(main())