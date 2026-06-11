# 2. Full AI-based parsing (LayoutParser) despite the laptop/offline constraint

Date: 2026-06-08

## Status

Accepted — **amended 2026-06-11** (Slice 3): the Detectron2 risk below materialized on
Apple Silicon, so the documented fallback was invoked. The AI-based branch now ships on
**Docling** rather than LayoutParser/Detectron2. See "Amendment" below.

## Context

The lecture presents two PDF-parsing approaches: **rule-based** (fast, light, breaks on varied
layouts) and **AI-based** (layout detection + OCR, handles messy/scanned documents, heavy and
slow). The lecture's AI-based exemplar is **LayoutParser**, which is built on Detectron2.

This project is **fully offline** and **laptop-sized**, and the corpus is an arbitrary
`documents/` folder. Two pragmatic alternatives exist:

- **PyMuPDF (rule-based)** — instant, light, but no OCR and weak on complex layouts. Sufficient
  if the folder is mostly digital-native PDFs.
- **Docling / Marker** — modern offline layout+OCR tools that install cleanly and run on CPU,
  achieving most of LayoutParser's benefit without Detectron2.

The project goal is to *faithfully illustrate the lecture's architecture*. The AI-based parsing
branch (layout detection → OCR → structured blocks, Figure 6) is a substantial part of the
lecture, and the structured-block output feeds our structure-aware chunking decision.

## Decision

Use **full AI-based parsing with LayoutParser**, verbatim to the lecture, accepting the setup
and runtime cost as the price of fidelity.

## Consequences

- The parsing stage demonstrates the lecture's AI-based branch exactly, and its structured-block
  output (titles, paragraphs, tables, reading order) directly enables structure-aware chunking.
- **Detectron2 has no official macOS / Apple-Silicon wheels.** On this Mac it typically must be
  built from source and often fails; budget real setup time or run the parsing stage in a Linux
  container / Colab. This is the main risk to the "runs on my laptop" experience.
- Parsing is slow on CPU. Combined with hash-based incremental ingestion, the cost is paid once
  per new/changed document rather than every run, which keeps it tolerable.
- If setup proves intractable, the fallback is Docling/Marker (still AI-based, offline) or a
  PyMuPDF-with-OCR-fallback hybrid — a contained change isolated to the parsing module.

## Amendment (2026-06-11, Slice 3)

Detectron2 could not be installed on the target Apple-Silicon Mac, exactly as the risk above
predicted. We invoked the documented fallback: the AI-based parsing branch now runs on
**Docling** (offline layout+OCR, CPU, installs cleanly), not LayoutParser/Detectron2.

What stayed faithful: the architecture is unchanged. The lecture's branch — *layout detection →
OCR → structured blocks (titles, paragraphs, tables, reading order)* — is realized intact; only
the model behind it changed. To keep that swap contained and the logic testable, detection sits
behind a `LayoutDetector` seam returning raw `DetectedRegion`s, and a deep `AiLayoutParser` maps
those onto `LayoutBlock`s (label→kind, column-aware reading order, source stamping). The parser
is unit-tested with a fake detector; the Docling adapter is heavy/non-deterministic and verified
manually (`RAG_PARSER=ai`), mirroring how the Ollama adapters are treated.

LayoutParser/Detectron2 remains the lecture-verbatim option for anyone on Linux/CUDA: drop in an
alternate `LayoutDetector` implementation, no downstream changes. The CLIP image branch is still
a documented stub, untouched here.
