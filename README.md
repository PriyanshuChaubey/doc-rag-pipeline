# RAG Pipeline — Document Question Answering System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built from scratch that enables any Large Language Model to answer questions grounded in your own documents — eliminating hallucinations by anchoring responses to real, retrieved content.

---

## The Problem

Large Language Models are powerful, but they have a fundamental limitation: **they only know what they were trained on**.

This creates three critical issues in real-world deployments:

- **Knowledge cutoff** — LLMs cannot answer questions about documents, reports, or data that postdate their training.
- **Hallucination** — When asked about domain-specific or private knowledge, LLMs confidently fabricate answers with no factual grounding.
- **No access to private data** — Enterprise documents, research papers, and internal knowledge bases are completely invisible to a general-purpose LLM.

The result: you cannot simply ask a model "What does this research paper say?" or "Summarise our internal report" and trust the answer.

---

## The Solution — RAG Architecture

This project implements a two-stage pipeline that solves the above problems by first **indexing** your documents into a searchable vector store, then **retrieving** only the most relevant content at query time and passing it as context to the LLM.

```
┌─────────────────────────────── Indexing Pipeline ──────────────────────────────┐
│                                                                                 │
│   PDF Files  ──►  Text Chunks  ──►  Embeddings  ──►  ChromaDB (Vector Store)  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                    (stored once)
                                          │
┌─────────────────────────────── Retrieval Pipeline ─────────────────────────────┐
│                                          │                                      │
│   User Query  ──►  Retriever  ◄──────────┘                                     │
│                        │                                                        │
│                   Top-K Chunks                                                  │
│                        │                                                        │
│                   Qwen3-32B (via Groq)  ──►  Grounded Answer                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The key insight: the LLM never guesses. It only synthesises an answer from chunks that were mathematically proven to be relevant to the query.

---

## Implementation Breakdown

### 1. Document Ingestion
Raw PDF files are loaded using LangChain's `PyPDFLoader`, which parses each page into a structured `Document` object containing the text content and its metadata (source file, page number). This forms the raw material for the entire pipeline.

### 2. Chunking Strategy
Long documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and an overlap of 50 characters. The overlap is deliberate — it ensures that context is not lost at chunk boundaries, so a sentence split across two chunks remains semantically intact in at least one of them.

### 3. Semantic Embedding
Each chunk is converted into a 384-dimensional dense vector using the `all-MiniLM-L6-v2` model from SentenceTransformers. These vectors capture the **semantic meaning** of the text — not just keywords — so that "What is encoder-decoder?" and "explain sequence-to-sequence architecture" map to nearby points in vector space.

### 4. Vector Storage with Cosine Similarity
Embeddings are persisted to disk using ChromaDB, configured explicitly with **cosine similarity** as the distance metric (`hnsw:space: cosine`). This is a critical design choice — cosine similarity measures the angle between two vectors, making it robust to differences in text length, whereas the default L2 (Euclidean) distance conflates direction with magnitude and produces less meaningful rankings for text retrieval.

### 5. Retrieval
The `RAGRetriever` class embeds the user's query using the same model and performs a nearest-neighbour search in ChromaDB, returning the top-K most semantically similar chunks. The similarity score is derived from the cosine distance as `1 - distance`, giving a normalised relevance score between 0 and 1.

### 6. Generation
The retrieved chunks are concatenated into a context string and injected into a prompt alongside the user query. This prompt is sent to **Qwen3-32B** hosted on Groq's low-latency inference API. The LLM generates an answer constrained by the provided context — grounding every response in your actual documents.

---

## Tech Stack

| Component | Tool / Library | Purpose |
|---|---|---|
| Document loading | `LangChain` + `PyPDFLoader` | Parse PDFs into Document objects |
| Text splitting | `RecursiveCharacterTextSplitter` | Chunking with boundary-safe overlap |
| Embeddings | `SentenceTransformers` — `all-MiniLM-L6-v2` | Semantic vector representation |
| Vector store | `ChromaDB` (persistent, cosine similarity) | Efficient nearest-neighbour retrieval |
| LLM | `Qwen3-32B` via `Groq` API | Context-aware answer generation |
| Orchestration | `LangChain Core` | Document and chain abstractions |

## Future Improvements

- Add multi-turn conversation memory to support follow-up questions
- Implement re-ranking (cross-encoder) on retrieved chunks for higher precision
- Support additional document formats — DOCX, web pages, Markdown
- Add a Gradio or Streamlit frontend for non-technical users
- Evaluate retrieval quality using RAGAS framework
