# Context

A glossary for the local RAG project — a scaled-down, fully-offline reimplementation of
the "ChatPDF" architecture from the lecture *06 - Retrieval-Augmented Generation*. The system
answers natural-language questions over a user-supplied set of PDFs.

## Glossary

### Document
A single PDF file placed in the `documents/` folder. The folder is the sole source of the
knowledge base; ingestion is folder-driven and generic (no fixed dataset). Replaces the
lecture's "company Wiki page / forum post."

### Knowledge base
The full set of [[Document]]s currently in the `documents/` folder, after parsing, chunking,
and indexing. The thing the system retrieves from.

### LayoutBlock
A structural unit of a parsed [[Document]] — a heading, a body paragraph, or a table — carrying
its source document and page. The quick-path parser emits everything as a `body` block; the
AI-based parser (Slice 8) fills in real headings and tables. The input to chunking.

### Chunk
A contiguous piece of a parsed [[Document]] small enough to embed and to fit the LLM context
window. The unit that gets embedded and indexed. Produced from [[LayoutBlock]]s by structure-aware
chunking: a heading is kept with its following body, a table is kept intact, blocks on different
pages are never merged (so [[Citation]]s stay page-accurate), and oversized content is split to a
token budget with overlap.

### Index
The local vector store holding one embedding per [[Chunk]], queried by nearest-neighbour
search during retrieval.

### Query
The user's natural-language question, embedded with the same model used for [[Chunk]]s and
compared against the [[Index]]. Before embedding it passes through [[Safety filtering]] and
[[Query expansion]].

### Query expansion
A single LLM rewrite of the raw [[Query]] — fixes typos, smooths phrasing, adds related
terms — performed before embedding to improve retrieval. Lightweight (one local-LLM call).

### Safety filtering
A lightweight check applied to the [[Query]] (input) and the final answer (output) for
disallowed content. Deliberately minimal per the lecture's deprioritization of safety.

### Text encoder
The model that turns text into embeddings (`nomic-embed-text`, run via Ollama). Note: this
project deliberately uses a dedicated sentence-embedding model **instead of** the lecture's
CLIP-as-text-encoder, because CLIP's 77-token limit truncates prose [[Chunk]]s. The CLIP-based
image branch is kept in the architecture as a documented stub, not implemented in v1.

### Citation
A bracketed reference (e.g. `[1]`) in the generated answer pointing back to the source
[[Chunk]] (and its document + page). Mandatory per the requirements ("include document
references" = Yes).

### Faithfulness
Whether the generated answer is grounded in the retrieved [[Chunk]]s rather than hallucinated.
One of the two aspects scored by the v1 LLM-as-judge evaluation (alongside answer relevance).
