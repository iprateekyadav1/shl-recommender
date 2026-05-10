"""
Conversational SHL assessment advisor agent.

Flow per call:
  1. Semantic search on the latest user message → top-10 catalog items.
  2. Build system prompt injecting the retrieved context.
  3. Call Groq (primary) or Gemini (fallback) LLM with full message history.
  4. Parse structured JSON from LLM reply.
  5. Return ChatResponse.
"""

import json
import logging
import os
import re
from typing import List, Optional, Tuple

from models import ChatResponse, Message, Recommendation
from retrieval import CatalogRetriever

logger = logging.getLogger(__name__)

MAX_TURNS = 8  # combined user + assistant turns


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are an expert SHL assessment advisor helping hiring managers choose the right \
individual test solutions for their open roles.

## Catalog context (retrieved for this query)
The following assessments from SHL's Individual Test Solutions catalog are \
potentially relevant. Use ONLY these items when making recommendations. \
Never invent assessment names or URLs.

{catalog_context}

## Your behaviour rules
1. CLARIFY first if the hiring need is too vague (no role, no seniority, no key skills).
   Ask at most 2 targeted questions per turn. Do NOT recommend on the very first turn
   for vague queries like "I need an assessment".
2. RECOMMEND 1–10 assessments once you have enough context (role, level, key competencies).
3. REFINE when the user adds constraints ("add personality", "remove coding tests") —
   update the shortlist without restarting the conversation.
4. COMPARE assessments when asked, using only catalog data provided. Never guess features.
5. Respect the 8-turn limit. If turn count is approaching the limit, commit to a
   recommendation even with partial information.
6. Set end_of_conversation to true ONLY after you have given recommendations AND the
   user seems satisfied (e.g., "thanks", "that's all", "looks good").

## Output format (STRICT — machine-parsed)
Always respond with a single valid JSON object. No prose outside the JSON. No markdown fences.

{{
  "reply": "<your conversational response to the user>",
  "recommendations": [
    {{
      "name": "<exact assessment name from catalog>",
      "url": "<exact URL from catalog>",
      "description": "<1–2 sentence explanation of why this fits the hiring need>"
    }}
  ],
  "end_of_conversation": false
}}

Rules:
- "recommendations" is [] when still clarifying.
- "recommendations" has 1–10 items when you are ready to commit to a shortlist.
- "end_of_conversation" is true only after a shortlist has been given AND the user is done.
- Strings inside JSON must not contain unescaped double-quotes.
"""


def _build_catalog_context(items: List[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        tags = ", ".join(item.get("tags", [])) or "general"
        lines.append(
            f"{i}. **{item['name']}**\n"
            f"   URL: {item['url']}\n"
            f"   Tags: {tags}\n"
            f"   Description: {item['description']}\n"
        )
    return "\n".join(lines)


def _count_turns(messages: List[Message]) -> int:
    return len(messages)


def _last_user_message(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """
    Try to parse a JSON object from LLM output.
    Handles:
      - raw JSON
      - JSON wrapped in ```json ... ```
      - JSON with a leading/trailing explanation sentence
    """
    text = text.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { … } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _parse_response(raw: str, catalog_items: List[dict]) -> ChatResponse:
    """Convert raw LLM text into a ChatResponse, falling back gracefully."""
    data = _extract_json(raw)

    if not data:
        # LLM returned non-JSON; treat whole reply as text, no recommendations
        logger.warning("LLM returned non-JSON response; wrapping as plain reply.")
        return ChatResponse(reply=raw[:2000], recommendations=[], end_of_conversation=False)

    reply = str(data.get("reply", "")).strip() or raw[:500]
    end_flag = bool(data.get("end_of_conversation", False))

    raw_recs = data.get("recommendations", [])
    recommendations: List[Recommendation] = []

    # Build a URL→item lookup from retrieved catalog items for validation
    valid_urls = {item["url"] for item in catalog_items}
    valid_names = {item["name"].lower(): item for item in catalog_items}

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        desc = str(rec.get("description", "")).strip()

        # Validate URL is from catalog; if not, try to find by name
        if url not in valid_urls:
            matched = valid_names.get(name.lower())
            if matched:
                url = matched["url"]
            else:
                logger.warning("Dropping hallucinated recommendation: %s / %s", name, url)
                continue

        if name and url:
            recommendations.append(Recommendation(name=name, url=url, description=desc))

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_flag,
    )


# ── LLM callers ──────────────────────────────────────────────────────────────

def _call_groq(system_prompt: str, messages: List[Message]) -> str:
    from groq import Groq  # type: ignore[import]

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    groq_messages = [{"role": m.role, "content": m.content} for m in messages]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + groq_messages,
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_gemini(system_prompt: str, messages: List[Message]) -> str:
    import google.generativeai as genai  # type: ignore[import]

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt,
    )
    # Build Gemini-compatible history
    history = []
    for msg in messages[:-1]:
        history.append(
            {"role": "user" if msg.role == "user" else "model", "parts": [msg.content]}
        )
    chat = model.start_chat(history=history)
    last = messages[-1].content if messages else ""
    response = chat.send_message(last)
    return response.text


def _call_llm(system_prompt: str, messages: List[Message]) -> str:
    """Try Groq first, fall back to Gemini."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if groq_key:
        try:
            return _call_groq(system_prompt, messages)
        except Exception as exc:
            logger.warning("Groq call failed: %s — trying Gemini…", exc)

    if gemini_key:
        try:
            return _call_gemini(system_prompt, messages)
        except Exception as exc:
            logger.error("Gemini call also failed: %s", exc)
            raise

    raise RuntimeError(
        "No LLM API key configured. Set GROQ_API_KEY or GEMINI_API_KEY."
    )


# ── Public interface ──────────────────────────────────────────────────────────

def run(messages: List[Message], retriever: CatalogRetriever) -> ChatResponse:
    """
    Main agent entry point. Stateless — takes full conversation history.
    """
    if not messages:
        return ChatResponse(
            reply="Hello! I'm your SHL assessment advisor. Tell me about the role you're hiring for.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Enforce turn limit
    turn_count = _count_turns(messages)
    if turn_count > MAX_TURNS:
        return ChatResponse(
            reply=(
                "We've reached the maximum conversation length. "
                "Based on our discussion, please review the recommendations above. "
                "Start a new conversation if you need further help."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    # Semantic search using the latest user message
    query = _last_user_message(messages)
    catalog_items = retriever.search(query, top_k=10)

    if not catalog_items:
        return ChatResponse(
            reply=(
                "I couldn't find relevant assessments in the catalog for that query. "
                "Could you describe the role or skills you need to assess in more detail?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    catalog_context = _build_catalog_context(catalog_items)
    system_prompt = _SYSTEM_TEMPLATE.format(
        catalog_context=catalog_context,
        turns_remaining=MAX_TURNS - turn_count,
    )

    raw_reply = _call_llm(system_prompt, messages)
    logger.debug("Raw LLM reply: %s", raw_reply[:500])

    return _parse_response(raw_reply, catalog_items)
