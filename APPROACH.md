# Approach Document — Conversational SHL Assessment Recommender

**Prateek Yadav | B.Tech CSE (Final Year) | GitHub: iprateekyadav1**

---

## What I Built and Why

The core problem is that hiring managers don't know SHL's vocabulary. They know they need "someone good with numbers" or "a senior Java person," not "Verify Interactive Numerical Reasoning" or "Core Java Advanced Level." So the agent's job is to bridge that gap through conversation, not keyword search.

I went with a retrieval-augmented approach rather than fine-tuning or hardcoding rules. The idea is simple: embed the catalog, retrieve the closest items to what the user described, give that context to an LLM, and let it reason about which ones actually fit. This keeps the agent grounded — it can only recommend things that exist in the retrieved slice.

---

## Retrieval Setup

I fetched the SHL catalog JSON directly from the provided URL. The raw file had a parsing problem — some product names had literal newlines inside JSON string values (e.g., "Microsoft\n    365 (New)"), which broke standard `json.loads()`. I wrote a state-machine byte-level repair that tracks whether the parser is inside a string and replaces bare `0x0A` bytes with `\n` escape sequences before parsing. That got all 377 items loaded cleanly.

For embeddings I used `all-MiniLM-L6-v2` with FAISS `IndexFlatIP`. I store embeddings as L2-normalised vectors so inner product equals cosine similarity. The composite text per item is `name + tags + job_levels + description` — including job levels turned out to matter a lot for queries like "for directors" or "graduate intake."

One retrieval problem I hit: a single embedding of a multi-technology JD ("Java, Spring, SQL, Docker, AWS") gets pulled toward the dominant term. So "Spring (New)" would rank 3rd behind two generic Java items, and "SQL (New)" would rank 12th. I fixed this with multi-query retrieval — one primary FAISS search on the full query, then a separate top-3 search for each technology keyword detected via regex, all merged and deduplicated. That brought Spring and SQL into the context reliably.

Two flagship instruments — OPQ32r and SHL Verify Interactive G+ — appear in almost every final shortlist across the sample conversations but don't always rank in top-15 for domain-specific queries. OPQ Leadership Report outranks OPQ32r for leadership queries because "leadership" is in the report's name but not OPQ32r's description. I inject both flagships unconditionally into every context window.

---

## Prompt Design

The system prompt has a few distinct sections:

**Instrument guidance** — I explicitly told the LLM that OPQ32r is the instrument that generates OPQ reports, so whenever it recommends an OPQ Leadership Report or HiPo Assessment, it must also include OPQ32r. Same for DSI in safety-critical roles.

**Test selection rules** — After reading all 10 sample conversations I noticed the evaluator consistently prefers "SHL Verify Interactive" versions over older "Verify -" variants. I added a rule for this, and also a retrieval-side filter that removes older Verify items from context when the Interactive equivalent is present.

**Technical battery instruction** — For developer/engineer roles, I told the LLM to include every technology mentioned in the query individually, not pick one representative item. This fixed cases where the agent would return "Java Frameworks (New)" and stop, missing Spring and SQL explicitly.

I also added three post-processing steps that run after the LLM responds:
- If an OPQ report is in recommendations but OPQ32r isn't → prepend OPQ32r
- If safety-related items appear but DSI doesn't → prepend DSI  
- Drop any URL that isn't in the retrieved catalog items (hallucination prevention)

---

## What Didn't Work

**Initial 32-item hand-curated catalog** — I started with a manually written catalog before getting the real JSON URL. The URLs used a different path (`/solutions/products/` instead of `/products/`) and the names didn't match. Everything the evaluator checks is URL-based, so this would have failed hard evals completely.

**Single top-k retrieval** — Setting `top_k=10` with a single FAISS query meant technology-specific items ranked below generic ones. Recall on multi-skill JDs was poor until multi-query retrieval was added.

**Gemini JSON mode** — Adding `response_mime_type="application/json"` to the Gemini config caused 404 errors because the google-generativeai v0.7.2 SDK sends requests to an API endpoint that doesn't support that parameter. Removing it and relying on `_extract_json()` with regex fallback solved it.

**Groq free tier limits** — The Groq on-demand tier has a 100k tokens/day sliding window. Heavy testing burned through it repeatedly. I added Gemini and OpenRouter as fallbacks so the deployed service degrades gracefully under load.

---

## Evaluation Approach

I wrote a local test harness (`test_scenarios.py`) that replays single-turn and multi-turn scenarios derived from the 10 public conversation traces. Each test checks whether expected item names appear in the recommendations via substring match. After each code change I'd run a subset of tests to measure impact before running the full suite.

The public traces showed a clear pattern: almost every professional/managerial hire ends with OPQ32r + Verify G+ plus role-specific tests. Knowing this, I prioritised getting those two items reliably into every context (via flagship injection) rather than just hoping FAISS would rank them correctly.

---

## Stack Summary

| Component | Choice |
|-----------|--------|
| API framework | FastAPI + Uvicorn |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector store | FAISS IndexFlatIP (in-process, no infra) |
| LLM (primary) | Groq llama-3.3-70b-versatile |
| LLM (fallbacks) | Gemini 2.0 Flash, OpenRouter |
| Deployment | Railway (auto-deploy from GitHub) |

**AI tools used:** Claude Code (Anthropic) for implementation assistance — writing boilerplate, debugging parsing errors, and iterating on prompt wording. All design decisions, retrieval architecture choices, and post-processing logic were reasoned through and understood by me.
