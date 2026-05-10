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
from fastapi.responses import HTMLResponse

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import agent
import catalog as catalog_module
from models import ChatRequest, ChatResponse
from retrieval import CatalogRetriever

_retriever: CatalogRetriever | None = None

_CHAT_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SHL Assessment Advisor</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#0f172a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
  header{background:#1e293b;border-bottom:1px solid #334155;padding:1rem 1.5rem;
         display:flex;align-items:center;gap:.75rem;flex-shrink:0}
  .logo{font-size:1.4rem;font-weight:800;color:#10b981;letter-spacing:-.02em}
  .sub{font-size:.85rem;color:#64748b}
  #chat{flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1rem}
  .bwrap{display:flex;flex-direction:column;max-width:78%}
  .bwrap.user{align-self:flex-end;align-items:flex-end}
  .bwrap.assistant{align-self:flex-start;align-items:flex-start}
  .bubble{padding:.75rem 1rem;border-radius:16px;line-height:1.6;font-size:.92rem;word-break:break-word}
  .bubble.user{background:#6366f1;color:#fff;border-bottom-right-radius:4px}
  .bubble.assistant{background:#1e293b;border:1px solid #334155;border-bottom-left-radius:4px}
  .lbl{font-size:.72rem;color:#64748b;margin-bottom:.3rem;padding:0 .25rem}
  .recs{margin-top:.75rem;display:flex;flex-direction:column;gap:.6rem;width:100%;max-width:520px}
  .card{background:#0f172a;border:1px solid #334155;border-radius:12px;padding:.85rem 1rem;
        display:flex;flex-direction:column;gap:.35rem;transition:border-color .15s}
  .card:hover{border-color:#6366f1}
  .cname{font-weight:600;font-size:.9rem}
  .crow{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
  .badge{font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:999px;letter-spacing:.04em}
  .A{background:#1e3a5f;color:#60a5fa}
  .B{background:#3b1f5e;color:#c084fc}
  .P{background:#1e3a5f;color:#818cf8}
  .K{background:#064e3b;color:#34d399}
  .S{background:#4a1942;color:#f472b6}
  .C{background:#3b2f00;color:#fbbf24}
  .E{background:#1c2e1c;color:#86efac}
  .clbl{font-size:.75rem;color:#64748b}
  .clink{font-size:.78rem;color:#6366f1;text-decoration:none}
  .clink:hover{text-decoration:underline}
  form{background:#1e293b;border-top:1px solid #334155;padding:1rem 1.5rem;
       display:flex;gap:.75rem;flex-shrink:0}
  textarea{flex:1;background:#0f172a;border:1px solid #334155;border-radius:10px;
           color:#e2e8f0;padding:.75rem 1rem;font-size:.92rem;resize:none;
           font-family:inherit;line-height:1.5;max-height:120px;overflow-y:auto}
  textarea:focus{outline:none;border-color:#6366f1}
  textarea::placeholder{color:#475569}
  button{background:#6366f1;color:#fff;border:none;border-radius:10px;
         padding:.75rem 1.4rem;font-weight:600;font-size:.9rem;cursor:pointer;white-space:nowrap;align-self:flex-end}
  button:hover{background:#4f46e5}
  button:disabled{opacity:.5;cursor:not-allowed}
  .typing{display:flex;gap:5px;align-items:center;padding:.6rem .75rem}
  .dot{width:7px;height:7px;background:#64748b;border-radius:50%;animation:bounce .9s infinite}
  .dot:nth-child(2){animation-delay:.15s}
  .dot:nth-child(3){animation-delay:.3s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
  .sysmsg{text-align:center;font-size:.78rem;color:#475569;padding:.25rem 0}
</style>
</head>
<body>
<header>
  <span class="logo">SHL.</span>
  <div>
    <div style="font-weight:600;font-size:.95rem">Assessment Advisor</div>
    <div class="sub">Conversational SHL Individual Test Solutions recommender</div>
  </div>
</header>
<div id="chat">
  <div class="sysmsg">Describe the role you are hiring for, or paste a job description.</div>
</div>
<form id="form">
  <textarea id="inp" rows="1" placeholder="e.g. I need to hire a mid-level Java developer who works with stakeholders…"
            oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px'"></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
var TT_LABELS = {
  A:"Ability & Aptitude", B:"Behavioural / SJT", C:"Competency",
  E:"Exercise", K:"Knowledge & Skills", P:"Personality", S:"Simulation"
};
var history = [];
var chat = document.getElementById('chat');
var inp  = document.getElementById('inp');
var send = document.getElementById('send');

function scrollBottom(){ chat.scrollTop = chat.scrollHeight; }

function mkEl(tag, cls, text){
  var el = document.createElement(tag);
  if(cls) el.className = cls;
  if(text !== undefined) el.textContent = text;
  return el;
}

function mkRecCard(r){
  var tt = (r.test_type || '').toUpperCase().charAt(0);
  var card = mkEl('div','card');
  card.appendChild(mkEl('div','cname', r.name));
  var row = mkEl('div','crow');
  var badge = mkEl('span','badge ' + (tt || 'A'), tt || '?');
  row.appendChild(badge);
  row.appendChild(mkEl('span','clbl', TT_LABELS[tt] || tt));
  card.appendChild(row);
  var a = document.createElement('a');
  a.className = 'clink';
  a.href = r.url;
  a.target = '_blank';
  a.rel = 'noopener noreferrer';
  a.textContent = r.url;
  card.appendChild(a);
  return card;
}

function appendBubble(role, text, recs){
  var wrap = mkEl('div','bwrap ' + role);
  wrap.appendChild(mkEl('div','lbl', role === 'user' ? 'You' : 'SHL Advisor'));
  wrap.appendChild(mkEl('div','bubble ' + role, text));
  if(recs && recs.length){
    var recsDiv = mkEl('div','recs');
    recs.forEach(function(r){ recsDiv.appendChild(mkRecCard(r)); });
    wrap.appendChild(recsDiv);
  }
  chat.appendChild(wrap);
  scrollBottom();
}

function addTyping(){
  var wrap = mkEl('div','bwrap assistant'); wrap.id = 'typing';
  wrap.appendChild(mkEl('div','lbl','SHL Advisor'));
  var bub = mkEl('div','bubble assistant typing');
  [0,1,2].forEach(function(){ bub.appendChild(mkEl('div','dot')); });
  wrap.appendChild(bub);
  chat.appendChild(wrap); scrollBottom();
}
function removeTyping(){ var t=document.getElementById('typing'); if(t) t.remove(); }

document.getElementById('form').addEventListener('submit', function(e){
  e.preventDefault();
  var text = inp.value.trim();
  if(!text) return;
  inp.value = ''; inp.style.height = 'auto';
  send.disabled = true;
  history.push({role:'user', content:text});
  appendBubble('user', text, null);
  addTyping();
  fetch('/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({messages: history})
  }).then(function(res){
    return res.json().then(function(data){ return {ok:res.ok, data:data}; });
  }).then(function(r){
    removeTyping();
    if(!r.ok){
      appendBubble('assistant', 'Error: ' + (r.data.detail || 'Unknown error'), null);
    } else {
      history.push({role:'assistant', content:r.data.reply});
      appendBubble('assistant', r.data.reply, r.data.recommendations);
      if(r.data.end_of_conversation){
        var m = mkEl('div','sysmsg');
        m.textContent = '— Conversation complete. Refresh to start a new one. —';
        chat.appendChild(m); scrollBottom();
        return;
      }
    }
    send.disabled = false; inp.focus();
  }).catch(function(){
    removeTyping();
    appendBubble('assistant','Network error. Is the server running?',null);
    send.disabled = false;
  });
});

inp.addEventListener('keydown', function(e){
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); document.getElementById('form').requestSubmit(); }
});
</script>
</body>
</html>"""


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
    title="SHL Assessment Recommender",
    version="1.0.0",
    description="Conversational agent for SHL Individual Test Solutions.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(content=_CHAT_UI)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Service initialising — retry in a few seconds.")
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="No LLM API key configured.")
    try:
        return agent.run(request.messages, _retriever)
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
