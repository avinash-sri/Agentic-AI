# Module 6 — Memory Deep Dive

## Why naive memory (just keep appending) breaks

1. **Context window limits** — every model has a max token count; long conversations eventually fail to fit.
2. **Cost scales with history size** — every call resends the entire conversation so far.
3. **"Lost in the middle"** — models are measurably worse at recalling info buried in the middle of a long context vs. the beginning/end.

## Strategy 1: Sliding Window

Keep only the last N messages, drop the rest.

```python
def trim_history(messages, max_messages=20, keep_system=True):
    system_msgs = [m for m in messages if m.get("role") == "system"] if keep_system else []
    other_msgs = [m for m in messages if m.get("role") != "system"]
    return system_msgs + other_msgs[-max_messages:]
```

Tradeoff: simple, but anything outside the window is **gone forever** — early context (e.g. destination city mentioned 25 messages ago) can be lost.

## Strategy 2: Summarization

Compress old messages into a summary via an LLM call, keep the summary instead of raw messages.

```python
def summarize_and_compress(messages, keep_recent=6):
    system_msg = messages[0]
    old_messages = messages[1:-keep_recent]
    recent_messages = messages[-keep_recent:]
    if not old_messages:
        return messages

    summary_prompt = "Summarize the key facts and decisions from this conversation:\n\n"
    summary_prompt += "\n".join(f"{m['role']}: {m.get('content','')}" for m in old_messages if m.get('content'))
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": summary_prompt}])
    summary = response.choices[0].message.content

    return [system_msg, {"role": "system", "content": f"Earlier conversation summary: {summary}"}] + recent_messages
```

Tradeoff: preserves the gist indefinitely at low token cost, but loses precise details (exact numbers/wording); costs an extra LLM call per compression.

## Strategy 3: Long-term memory via retrieval (RAG-style)

For memory that should persist **across separate sessions** (not just within one conversation).

### Core idea
1. Store facts as text, convert each to an **embedding** (vector representing meaning)
2. Embed the new incoming message too
3. Find stored memories with the most *similar* embeddings (cosine similarity)
4. Inject only the relevant ones into the prompt

```python
import numpy as np

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # dot product measures alignment; dividing by both norms removes magnitude,
    # leaving pure "direction" (semantic similarity), range -1 to 1

memory_store = []  # [{"text": ..., "embedding": ...}, ...]

def save_memory(text):
    memory_store.append({"text": text, "embedding": get_embedding(text)})

def recall_relevant_memories(query, top_k=3):
    if not memory_store:
        return []
    query_embedding = get_embedding(query)
    scored = [(m["text"], cosine_similarity(query_embedding, m["embedding"])) for m in memory_store]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, score in scored[:top_k]]
```

Usage — inject recalled memories into the system prompt before the model sees the new message:
```python
relevant = recall_relevant_memories("Plan me a trip to Goa")
context = "\n".join(relevant)
messages = [
    {"role": "system", "content": f"Relevant facts about this user:\n{context}"},
    {"role": "user", "content": "Plan me a trip to Goa"}
]
```

This is retrieval **by meaning, not exact keyword matching** — the same underlying idea behind RAG chatbots and "chat with your PDF" tools generally, not agent-specific.

Production: replace the in-memory list with a real vector DB (Chroma, Pinecone, Weaviate, pgvector) for persistence + fast search at scale (HNSW indexing).

## Realistic combined architecture

```
Long-term memory (vector DB)       ← persists across sessions
        ↓ (relevant facts injected)
Summarized older context           ← this session, older turns
        ↓
Recent raw messages (sliding window) ← this session, last N turns, full detail
        ↓
      Sent to the LLM
```

## Key Lessons

1. Memory design = an active decision about what to keep/compress/drop, not "append forever."
2. Cosine similarity measures **direction**, not magnitude — this matches semantic similarity better than raw distance for text embeddings.
3. Embedding provider is swappable — the region/API restriction issue below has nothing to do with the underlying math or your retrieval code.

## Bugs I Hit

- `openai.AuthenticationError: Attempted to access resource from outside project geography US` when calling `client.embeddings.create` — an **account/project-level geography restriction**, not a code bug. All OpenAI calls from that project are blocked outside the allowed region, regardless of model/endpoint.
  - **Fix used**: switched to a **local embedding model** via `sentence-transformers`, which runs entirely on-device with no API call, sidestepping the restriction entirely:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def get_embedding(text):
        return model.encode(text).tolist()
    ```
    Drop-in replacement — `cosine_similarity`, `save_memory`, `recall_relevant_memories` all unchanged since they just operate on lists of floats.
  - Verified end-to-end: saved "prefers beach destinations" in one session, started a fresh `History`, asked for a destination suggestion with no mention of beaches — got Sri Lanka (a beach destination) back, confirming retrieval genuinely influenced the answer.
 