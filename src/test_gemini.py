import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Sending request to Gemini...")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello and tell me one cool fact about AI."
)

print(response.text)