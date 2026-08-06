from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# This list IS the memory. We keep appending to it.
conversation = [
    {"role": "system", "content": "You are a friendly, concise assistant."}
]

def chat(user_input):
    # 1. Add the user's message to history
    conversation.append({"role": "user", "content": user_input})

    # 2. Send the ENTIRE history so far
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=conversation
    )

    # 3. Extract the assistant's reply
    reply = response.choices[0].message.content

    # 4. Add the assistant's reply to history too — otherwise it forgets what IT said
    conversation.append({"role": "assistant", "content": reply})

    return reply

print(chat("My name is Rahul."))
print(chat("What's my name?"))
print(chat("What have we talked about so far?"))
print(conversation) 
print(len(conversation)) # This is the memory — it contains the entire conversation so far