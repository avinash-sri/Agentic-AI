import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def load_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap keeps context from being cut mid-idea
    return chunks

document_text = load_pdf_text("trip_guide.pdf")
chunks = chunk_text(document_text)
# print(chunks[0])  # print the first chunk to verify
# print("-----------")
# print(chunks[1])
# print(f"Split into {len(chunks)} chunks")


model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text).tolist()

document_store = []  # [{"text": chunk, "embedding": [...]}, ...]

for chunk in chunks:
    document_store.append({"text": chunk, "embedding": get_embedding(chunk)})



def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = get_embedding(query)
    scored = [(c["text"], cosine_similarity(query_embedding, c["embedding"])) for c in document_store]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, score in scored[:top_k]]

def rag_answer(question):
    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so — don't make anything up.

                Context:
                {context}

                Question: {question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# print(rag_answer("What's the best time of year to visit Goa according to this guide?"))
print(rag_answer("how is sri lanka as a travel destination?")) # to check hallucinations