# SHL Conversational Assessment Recommender — Approach Document

## Overview

A stateless FastAPI service that converts vague hiring intent into grounded SHL
assessment shortlists through multi-turn dialogue. Built with sentence-transformer
semantic retrieval (FAISS), a system-prompted LLM, and deterministic post-processing
to enforce catalog grounding and key instrument coverage.

---

## Architecture

```
POST /chat  ──►  FAISS retrieval (top-15)  ──►  flagship injection
                                              ──►  LLM (Groq → Gemini → OpenRouter)
                                              ──►  JSON parse + URL validation
                                              ──►  OPQ32r post-processing
                                              ──►  ChatResponse
```

### Catalog ingestion

The full SHL product catalog JSON (377 items) was fetched from the provided data URL.
The raw JSON contained unescaped literal newlines inside string values (malformed JSON);
a state-machine byte-level repair was applied before parsing. Each catalog item was
enriched with:

- `test_type`: mapped from SHL's `keys` field to single/multi-letter codes
  (K = Knowledge & Skills, P = Personality, A = Ability, B = Situational Judgment,
  C = Competencies, S = Simulations, D = Development & 360, E = Exercises)
- `tags`: extracted from job levels, test categories, and name keyword matching
- `job_levels`, `duration`, `languages`: preserved for context and retrieval

### Retrieval

`sentence-transformers/all-MiniLM-L6-v2` generates 384-dim embeddings over a composite
field (name + tags + job_levels + description). Vectors are L2-normalised and stored in
a `faiss.IndexFlatIP` so inner-product equals cosine similarity. At query time:

1. Top-15 nearest catalog items are retrieved for the latest user message.
2. Two flagship instruments — OPQ32r and SHL Verify Interactive G+ — are unconditionally
   injected into the context if they didn't rank in the top 15. These items are relevant
   to almost every professional/managerial hire but can be crowded out by role-specific
   report products with stronger term overlap.

### LLM prompting

The system prompt structure:
- Mandatory instrument guidance (OPQ32r for personality, Verify G+ for cognitive)
- Test selection rules: prefer SHL Verify Interactive over older Verify tests; provide
  full technical batteries for developer/engineer roles
- Strict rules: clarify before recommending, refuse off-topic, honor turn cap
- JSON output format with recommendations validated against the injected catalog context

The LLM receives the 15–17-item catalog context (with duration, languages, job levels)
and outputs structured JSON. Groq `llama-3.3-70b-versatile` is primary; Google Gemini
and OpenRouter are fallbacks.

### URL hallucination prevention

Every LLM-returned URL is validated against the retrieved catalog items. Hallucinated
URLs are remapped by name match, or dropped if no match exists. test_type is backfilled
from catalog data when the LLM omits it.

### OPQ32r post-processing

If the LLM recommends any OPQ-branded report (OPQ Leadership Report, Enterprise
Leadership Report, HiPo, Sales Transformation, etc.) but does not include OPQ32r,
OPQ32r is prepended automatically. This is deterministic domain knowledge: all OPQ
reports are generated from a single OPQ32r administration.

---

## Evaluation

**Testing against sample conversations (C1–C10):**

The 10 provided traces cover leadership selection, technical developer hiring, contact
centre screening, graduate recruitment, safety-critical roles, and mixed bilingual
scenarios. After reading all traces, key patterns identified:

- OPQ32r appears in ~80% of final shortlists; retrieval alone was insufficient
- The evaluator runs realistic multi-turn LLM-simulated users, not fixed scripts
- Items recommended must use exact catalog URLs (no solutions/products URL variants)

**What didn't work:**

- Initial catalog (32 manually curated items) missed many real catalog products
  referenced in sample conversations (SVAR, DSI, Graduate Scenarios, etc.)
- `var history = []` in the chat UI conflicted with `window.history` — silent JS crash
- Groq's `llama3-70b-8192` model was decommissioned mid-development
- top_k=10 retrieval caused OPQ32r to rank below leadership-specific report products

**What improved things:**

- Replacing 32-item catalog with full 377-item real SHL JSON improved Recall significantly
- Flagship item injection (unconditional OPQ32r + Verify G+ in context) + post-processing
- Richer composite text (+ job_levels) improved role-level retrieval
- System prompt test selection rules reduced wrong-variant selection (Verify Interactive)

---

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Framework | FastAPI + Uvicorn | Async-ready, Pydantic v2 schema enforcement |
| Embeddings | all-MiniLM-L6-v2 | Fast CPU inference, 384-dim, good semantic recall |
| Vector store | FAISS IndexFlatIP | Zero infrastructure, exact cosine search |
| LLM primary | Groq llama-3.3-70b-versatile | Fast inference, JSON mode, free tier |
| LLM fallback | Gemini 1.5 Flash / OpenRouter | Rate-limit resilience |
| Catalog source | SHL catalog JSON (provided URL) | 377 items, scraped 2026-05-08 |

**AI tools used:** Claude Code (Anthropic) for implementation assistance throughout.
All design decisions, debugging, and architectural choices were made and understood
by the developer.
