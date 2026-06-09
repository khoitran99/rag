# 1. Dedicated sentence-embedding model instead of CLIP as the text encoder

Date: 2026-06-08

## Status

Accepted

## Context

The lecture *06 - Retrieval-Augmented Generation* uses a single **CLIP** model as both the
text encoder and the image encoder, so text and image embeddings share one space and can be
cross-retrieved. Following the lecture verbatim would mean embedding our text chunks with
CLIP's text tower.

CLIP's text encoder has a hard **77-token input limit**. Our chunking is structure-aware:
chunks are whole sections or intact tables, routinely far longer than 77 tokens. Embedding
those with CLIP would silently truncate most of each chunk, degrading retrieval quality —
the exact opposite of why we paid for layout-aware parsing and chunking.

v1 is also **text-only** (the image branch is a documented stub), so the cross-modal shared
space that justifies CLIP is not exercised yet.

## Decision

Use a dedicated sentence-embedding model — **`nomic-embed-text` via Ollama** (768-dim,
8192-token context) — as the text encoder, instead of CLIP. Keep the CLIP-based image
encoder / image index in the architecture diagram as a stub for a future multimodal phase.

## Consequences

- Large structure-aware chunks embed without truncation; retrieval reflects full chunk content.
- One runtime (Ollama) serves both embedding and generation — fewer dependencies, fully offline.
- The embedding dimension (768) is now baked into the FAISS index; changing the embedder later
  requires a full re-index, so this is costly to reverse.
- We deviate from the lecture on this one box. When the image branch is implemented, text–image
  alignment will need a separate strategy (e.g. CLIP for images + image-captioning into the
  text-embedding space, per the lecture's "Approach 2"), since text is no longer in CLIP space.
