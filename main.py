"""
SHL Conversational Assessment Recommender — FastAPI application entry point.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

# Resolve absolute path so it works regardless of uvicorn's CWD
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import agent
import catalog as catalog_module
from models import ChatRequest, ChatResponse
from retrieval import CatalogRetriever

# ── Global state (populated at startup) ───────────────────────────────────────
_retriever: CatalogRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    logger.info("Starting up — loading catalog and building FAISS index…")
    try:
        items = await catalog_module.build_catalog()
        _retriever = CatalogRetriever(items)
        logger.info("Ready. Catalog has %d items.", len(items))
    except Exception:
        logger.exception("Startup failed — service will be degraded.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    description="Conversational AI agent that recommends SHL Individual Test Solutions.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SHL Assessment Recommender</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 2rem;
    }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 3rem;
      max-width: 560px;
      width: 100%;
      text-align: center;
      box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    }
    .badge {
      display: inline-block;
      background: #10b981;
      color: #fff;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
      padding: 4px 12px;
      border-radius: 999px;
      margin-bottom: 1.5rem;
    }
    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: .75rem; }
    p  { color: #94a3b8; line-height: 1.7; margin-bottom: 2rem; font-size: .95rem; }
    .buttons { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
    a.btn {
      display: inline-block;
      padding: .65rem 1.4rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: .9rem;
      text-decoration: none;
      transition: opacity .15s;
    }
    a.btn:hover { opacity: .85; }
    .btn-primary { background: #6366f1; color: #fff; }
    .btn-secondary { background: #334155; color: #e2e8f0; }
    .endpoints {
      margin-top: 2.5rem;
      text-align: left;
      background: #0f172a;
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
    }
    .endpoints h3 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b; margin-bottom: .85rem; }
    .ep { display: flex; align-items: center; gap: .75rem; margin-bottom: .55rem; font-size: .88rem; }
    .method {
      font-size: .72rem; font-weight: 700; padding: 2px 7px;
      border-radius: 4px; letter-spacing: .04em; min-width: 42px; text-align: center;
    }
    .get  { background: #064e3b; color: #34d399; }
    .post { background: #1e3a5f; color: #60a5fa; }
    .ep-path { color: #cbd5e1; font-family: monospace; }
    .ep-desc { color: #64748b; font-size: .8rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">Live</div>
    <h1>SHL Assessment Recommender</h1>
    <p>A conversational AI agent that helps hiring managers find the right
       SHL Individual Test Solutions through natural dialogue.</p>
    <div class="buttons">
      <a class="btn btn-primary" href="/docs">Interactive API Docs</a>
      <a class="btn btn-secondary" href="/health">Health Check</a>
    </div>
    <div class="endpoints">
      <h3>Endpoints</h3>
      <div class="ep">
        <span class="method get">GET</span>
        <span class="ep-path">/health</span>
        <span class="ep-desc">Service status</span>
      </div>
      <div class="ep">
        <span class="method post">POST</span>
        <span class="ep-path">/chat</span>
        <span class="ep-desc">Conversational assessment advisor</span>
      </div>
      <div class="ep">
        <span class="method get">GET</span>
        <span class="ep-path">/docs</span>
        <span class="ep-desc">Swagger UI — try the API here</span>
      </div>
    </div>
  </div>
</body>
</html>
""")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Service is still initialising — please retry in a few seconds.",
        )

    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="No LLM API key configured. Set GROQ_API_KEY or GEMINI_API_KEY.",
        )

    try:
        return agent.run(request.messages, _retriever)
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
