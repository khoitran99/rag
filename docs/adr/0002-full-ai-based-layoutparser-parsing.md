# 2. Full AI-based parsing (LayoutParser) despite the laptop/offline constraint

Date: 2026-06-08

## Status

Accepted

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
