from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env automatically

client = OpenAI()  # it auto-detects the key from the environment

response = client.chat.completions.create(
    model="gpt-5-nano",   # a cheap, fast model — good for learning
    messages=[
        {"role": "system", "content": "You are a sarcastic pirate."},
        {"role": "user", "content": "Explain agentic AI in one sentence."}
    ]
)

print(response.choices[0].message.content)