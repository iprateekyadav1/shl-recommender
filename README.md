# SHL Conversational Assessment Recommender

A stateless FastAPI service that acts as a conversational AI agent for recommending SHL Individual Test Solutions. It guides hiring managers through a dialogue to find the right assessments for their open roles.

---

## Architecture

```
POST /chat  →  agent.run()
                  ├── CatalogRetriever.search()  (sentence-transformers + FAISS)
                  ├── LLM call  (Groq llama3-70b-8192  |  Gemini 1.5-flash)
                  └── JSON parse + URL validation
```

- **Stateless**: Every `/chat` call receives the full conversation history. No server-side session storage.
- **Catalog**: 32 SHL Individual Test Solutions stored in `catalog.json`, embedded with `all-MiniLM-L6-v2`, indexed in FAISS.
- **LLM**: Groq (primary) → Gemini (fallback). Structured JSON mode enforced.

---

## Running Locally

### 1. Clone / copy the project

```bash
git clone <your-repo>
cd shl-recommender
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` and `faiss-cpu` are heavy (~1 GB total). First install may take a few minutes.

### 4. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY (get a free key at console.groq.com)
```

### 5. Start the server

```bash
uvicorn main:app --reload
```

Server starts at `http://localhost:8000`. On first start it loads `catalog.json` and builds the FAISS index (takes ~10-20 seconds).

---

## Required Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (or Gemini) | Groq API key — free at [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | Yes (or Groq) | Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com) |

At least one key must be set. The service tries Groq first and falls back to Gemini.

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a mid-level Java developer who also works with stakeholders"}
    ]
  }'
```

**Response:**
```json
{
  "reply": "Great, I can help with that. To narrow down the best assessments, could you tell me how many years of experience qualifies as mid-level for your team, and are stakeholder skills equally important to technical Java skills?",
  "recommendations": [],
  "end_of_conversation": false
}
```

**Multi-turn example (full conversation):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
      {"role": "assistant", "content": "Sure. What is the seniority level?"},
      {"role": "user", "content": "Mid-level, around 4 years experience"}
    ]
  }'
```

**Response:**
```json
{
  "reply": "Got it — mid-level Java developer with stakeholder interaction. Here are 5 assessments that fit this profile.",
  "recommendations": [
    {
      "name": "Verify Interactive Java",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-java/",
      "description": "Directly tests Java skills across OOP, concurrency, and APIs at the right difficulty for a 4-year developer."
    },
    {
      "name": "Verify Verbal Reasoning",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/verbal-reasoning-1/",
      "description": "Stakeholder-facing roles need strong written and verbal comprehension — this validates that ability."
    },
    {
      "name": "OPQ32 (Occupational Personality Questionnaire)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
      "description": "Reveals teamwork style, communication approach, and influence skills essential for cross-functional work."
    }
  ],
  "end_of_conversation": false
}
```

---

## Deploying on Render.com

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### Step 2 — Create a new Web Service on Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repository
3. Render auto-detects `render.yaml` — click **Apply**

### Step 3 — Set environment variables

In the Render dashboard for your service → **Environment**:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `GEMINI_API_KEY` | (optional) your Gemini key |

### Step 4 — Deploy

Click **Manual Deploy** → **Deploy latest commit**. The first deploy takes 3–5 minutes (installing dependencies). The `/health` endpoint must respond within 2 minutes of cold start.

### Step 5 — Test production endpoint

```bash
curl https://<your-service>.onrender.com/health

curl -X POST https://<your-service>.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to assess senior Python engineers"}]}'
```

---

## Agent Behaviour

| Behaviour | When triggered |
|---|---|
| **Clarify** | Query is vague (no role, level, or skills). Agent asks ≤2 questions. |
| **Recommend** | Enough context gathered. Returns 1–10 assessments with real catalog URLs. |
| **Refine** | User adds/removes constraints. Shortlist updated without restart. |
| **Compare** | User asks about differences. Agent answers from catalog data only. |
| **End** | User satisfied after recommendations. `end_of_conversation: true`. |

Conversations are capped at **8 turns** (combined user + assistant).

---

## Project Structure

```
shl-recommender/
├── main.py           # FastAPI app, lifespan, endpoints
├── agent.py          # Conversational agent logic, LLM calls, JSON parsing
├── catalog.py        # Catalog scraping + fallback, catalog.json writer
├── retrieval.py      # SentenceTransformer + FAISS semantic search
├── models.py         # Pydantic request/response models
├── catalog.json      # Pre-populated SHL Individual Test Solutions (32 items)
├── requirements.txt
├── render.yaml       # Render.com deployment config
├── .env.example      # Template for local env vars
└── README.md
```
