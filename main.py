"""
SHL Conversational Assessment Recommender — FastAPI application entry point.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

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
