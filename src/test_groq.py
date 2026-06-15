import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("Sending request to Groq...")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Say hello and tell me one cool fact about AI."
        }
    ]
)

print(response.choices[0].message.content)