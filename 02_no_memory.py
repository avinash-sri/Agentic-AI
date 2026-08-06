from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response1 = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "My name is Rahul."}]
)
print("Turn 1:", response1.choices[0].message.content)

response2 = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "What's my name?"}]
)
print("Turn 2:", response2.choices[0].message.content)