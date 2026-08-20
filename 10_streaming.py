from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a short paragraph about Sri Lanka as a travel destination."}],
    stream=True  # <-- this one flag changes everything
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:  # some chunks carry no text (e.g., the very first/last chunk)
        print(delta, end="", flush=True)

print()  # final newline