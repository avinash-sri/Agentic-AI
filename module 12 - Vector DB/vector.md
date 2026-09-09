# Module 12 — Real Vector Database (Chroma)

## Why a real vector DB vs a Python list

| | Python list (Modules 6 & 10) | Chroma |
|---|---|---|
| Persistence | Gone when script ends | Saved to disk, survives restarts |
| Search method | Brute-force loop + manual cosine similarity | Indexed nearest-neighbor search |
| Scale | Fine for hundreds of items | Built for thousands–millions |
| Filtering | Write it yourself | Built-in metadata filtering (`where={...}`) |
| Embedding management | Call `get_embedding()` yourself every time | Can auto-embed internally |

## Setup

```bash
pip install chromadb
```

## Basic usage

```python
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # saves to disk
collection = chroma_client.get_or_create_collection(name="trip_memories")

def save_memory(text, memory_id):
    collection.add(documents=[text], ids=[memory_id])  # unique id per item

def recall_relevant_memories(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]
```

Chroma has a default embedding function built in — no manual `get_embedding()` call needed, unlike the Module 6 version.

## Using your own embedding model (consistency with Module 6/10)

```python
from chromadb.utils import embedding_functions

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="trip_memories", embedding_function=sentence_transformer_ef)
```

## Metadata filtering

```python
collection.add(
    documents=["Beach info for Goa", "Mountain trekking guide for Manali"],
    ids=["doc_1", "doc_2"],
    metadatas=[{"destination_type": "beach"}, {"destination_type": "mountain"}]
)

results = collection.query(query_texts=["best time to visit"], n_results=2, where={"destination_type": "beach"})
```

`where` filters before/alongside similarity search — the capability the raw Python list lacked without custom filtering logic.

## Rebuilding Module 10's RAG store with Chroma

```python
doc_collection = chroma_client.get_or_create_collection(name="trip_guide_docs")

def ingest_document(path):
    chunks = chunk_text(load_pdf_text(path))
    doc_collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])

def retrieve_relevant_chunks(query, top_k=3):
    results = doc_collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]
```

`rag_answer()` from Module 10 needs no changes — clean separation: the interface (query in, relevant text out) stayed the same; only the implementation underneath swapped from a list to Chroma.

## FAISS — a related but different tool (not "for OpenAI" specifically)

FAISS (Facebook AI Similarity Search) is a general-purpose vector similarity **library**, provider-agnostic — works with OpenAI embeddings, `sentence-transformers`, or anything else. Not tied to any one embedding provider.

| | Chroma | FAISS |
|---|---|---|
| What it is | Full vector *database* (storage + search + metadata + persistence) | *Library* purely for fast similarity search math |
| Persistence | Built in | Manual (you save/load the index yourself) |
| Metadata filtering | Built in | Not built in — manage a separate mapping yourself |
| Generates embeddings? | Can auto-embed | No — bring your own embeddings |
| Best for | Fast working system, less code | Very large scale / max speed, comfortable building surrounding infra |

Often, FAISS is what's running *underneath* other vector databases — it's the low-level search engine; Chroma is the friendly, batteries-included wrapper.

```python
import faiss
import numpy as np

dimension = 1536  # text-embedding-3-small output size
index = faiss.IndexFlatL2(dimension)

embeddings = np.array([get_openai_embedding(t) for t in texts]).astype("float32")
index.add(embeddings)

query_embedding = np.array([get_openai_embedding("Suggest a trip")]).astype("float32")
distances, indices = index.search(query_embedding, k=2)
for idx in indices[0]:
    print(texts[idx])  # you manage the index-position -> text mapping yourself
```

## Key Lessons

1. A vector database = persistence + indexed search + metadata filtering + (often) embedding management, all bundled — vs. a plain list where you build all of that yourself.
2. FAISS is a similarity-search library, not a database and not provider-specific — it can sit underneath any embedding source, including OpenAI's.
3. Good software design check: swapping storage backend (list → Chroma) required no changes to the calling code (`rag_answer`) — because the interface (function signature) stayed stable while the implementation changed.
4. Chroma is a solid default for learning and small-to-medium production; FAISS/Pinecone/Weaviate/pgvector matter more at larger scale or when you need lower-level control.

## To verify (persistence check)

Run one script that only saves data via `save_memory`/`ingest_document`, exit, then run a **second separate script** that only queries — confirm the data is still retrievable. This is the actual proof persistence works, vs. the in-memory list from Module 6 which loses everything when the script ends.
