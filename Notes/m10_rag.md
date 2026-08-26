# Module 10 — RAG over Documents

## Why RAG exists

Two ways to give an LLM knowledge it doesn't have:
1. **Stuff the whole document in the prompt** — breaks down for long documents (context limits, cost scales per question, "lost in the middle" from Module 6.1)
2. **RAG** — store the document as searchable chunks, retrieve only relevant chunks per question, inject just those into the prompt

RAG = Module 6's long-term memory strategy, applied to documents instead of user facts. **The generation step is identical to a basic Module 1 API call** — RAG's novelty lives entirely in what happens *before* that call (ingestion + retrieval), not in a new way of talking to the LLM.

## Two separate pipelines

```
INGESTION (runs ONCE per document)
Document → Load → Chunk → Embed → Store

RETRIEVAL (runs EVERY question)
Question → Embed → Search stored chunks → Top-k → Inject into prompt → LLM answers
```

They're separate because they run at different frequencies: ingestion is the expensive "build an index in advance" step (like a search engine crawling pages before anyone searches); retrieval is the fast per-query lookup against an already-built index.

### Ingestion

```python
from pypdf import PdfReader

def load_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap prevents cutting an idea mid-sentence
    return chunks

document_store = []
def ingest_document(path):
    text = load_pdf_text(path)
    for chunk in chunk_text(text):
        document_store.append({"text": chunk, "embedding": get_embedding(chunk)})
```

### Retrieval (reuses Module 6's `cosine_similarity` — no new math)

```python
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = get_embedding(query)
    scored = [(c["text"], cosine_similarity(query_embedding, c["embedding"])) for c in document_store]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, score in scored[:top_k]]
```

### Generation — grounding is the critical instruction

```python
def rag_answer(question):
    context = "\n\n---\n\n".join(retrieve_relevant_chunks(question))
    prompt = f"""Answer using ONLY the context below. If the answer isn't in the context, say so — don't make anything up.

Context:
{context}

Question: {question}"""
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content
```

"Using ONLY the context... say so if not present" is the single most important line in any RAG system — without it the model blends its own training knowledge with retrieved context, and answers are no longer verifiably grounded in your document. Same failure mode/fix as "trust the tool result over your own memory" (Module 2/3).

### As an agent tool

```python
def search_document(query):
    return "\n\n".join(retrieve_relevant_chunks(query))
```
Register like any other tool (Module 3) — the agent decides when to consult the document vs. other tools vs. its own knowledge.

## Practical test to run

Always ask at least one question whose answer is genuinely **not** in the document, to confirm it says "I don't know" instead of hallucinating.

## Further considerations for production RAG (roughly in order of impact)

1. **Chunk size/overlap tuning** — fixed character-count chunking can cut tables/sentences/code awkwardly. Semantic/recursive chunking (paragraph → sentence boundaries, e.g. `RecursiveCharacterTextSplitter`) is better. Too-small chunks lack context; too-large chunks dilute embedding precision and waste tokens.
2. **Reranking** — embedding similarity is a cheap, rough filter. Two-stage retrieval (embed+search top ~20 candidates → rerank with a slower, more precise cross-encoder → keep top 3) consistently improves quality.
3. **Hybrid search** — combine semantic (embedding) search with keyword/BM25 search; pure embeddings can miss exact matches (product codes, error messages, proper nouns).
4. **Metadata filtering** — store metadata (date, source, page) alongside each chunk to filter *before* similarity search (e.g., "only documents after 2024") — vector similarity alone can't express this.
5. **RAG-specific evaluation** — measure retrieval quality (is the right chunk in top-k?) and generation quality (given good chunks, is the answer correct?) *separately* — conflating them hides where a bad answer actually went wrong.
6. **Graceful "no answer" handling** — check the top similarity score itself; if below a threshold, skip the LLM call and return "I don't have information on that" directly, rather than tempting the model to stretch weak context into a plausible-but-wrong answer.
7. **Source attribution** — cite which document/page a claim came from, especially important for trust in legal/medical/research use cases.

## Key Lessons

1. Ingestion and retrieval are separate pipelines running at different frequencies — adding a new document only re-runs ingestion, never touches the retrieval/generation code.
2. RAG is a *prompt-construction strategy* (what text to inject), not a new LLM interaction pattern.
3. Grounding instructions are non-negotiable — without them, "RAG" degrades into the model just using its own training knowledge with extra irrelevant text attached.