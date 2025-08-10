from __future__ import annotations

import os
import time
import uuid
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_system import CreditCardRAG, Answer, required_profile_slots
from guardrails import sanitize_user_text
from rag_system import gpt_complete  # diagnostic


# ----------------------------------------------------------------------------
# App init
# ----------------------------------------------------------------------------
load_dotenv()

app = FastAPI(title="Credit Card Advisor API", default_response_class=ORJSONResponse)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost:3000", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Session store (in-memory)
# ----------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
    ts: int = Field(default_factory=lambda: int(time.time()))


class SessionData(BaseModel):
    chat: List[Message] = Field(default_factory=list)
    profile: Dict[str, Any] = Field(default_factory=lambda: {k: None for k in required_profile_slots()})
    last_cards: List[Dict[str, Any]] = Field(default_factory=list)


_SESSIONS: Dict[str, SessionData] = {}


def get_or_create_session(session_id: Optional[str]) -> str:
    sid = session_id or str(uuid.uuid4())[:8]
    if sid not in _SESSIONS:
        _SESSIONS[sid] = SessionData()
    return sid


# ----------------------------------------------------------------------------
# RAG singleton
# ----------------------------------------------------------------------------
_RAG: Optional[CreditCardRAG] = None


def get_rag(force_reindex: bool = False, data_path: Optional[str] = None) -> CreditCardRAG:
    global _RAG
    if _RAG is None or force_reindex or data_path:
        _RAG = CreditCardRAG(force_reindex=force_reindex, data_path=data_path)
    return _RAG


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    suggestions: List[str] = []
    profile: Dict[str, Any] = {}
    cards: List[Dict[str, Any]] = []


class UploadResponse(BaseModel):
    ok: bool
    message: str
    path: Optional[str] = None


# ----------------------------------------------------------------------------
# Quick prompts (predefined)
# ----------------------------------------------------------------------------
QUICK_PROMPTS = [
    "Recommend an SBI cashback card under ₹1000 with lounge access",
    "Compare HDFC Millennia vs SBI SimplyCLICK",
    "I’m a student with no credit history",
    "Self-employed, ₹80k/month, CIBIL 760 — premium options?",
    "Best fuel + groceries card under ₹500 fee",
]


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    rag = get_rag()
    return {"ok": True, "openai": bool(os.getenv("OPENAI_API_KEY")), "tavily": bool(os.getenv("TAVILY_API_KEY")), "dataset_rows": len(rag.retriever.df) if rag and rag.retriever else 0}


@app.get("/api/llm_diag")
def llm_diag():
    """Quick LLM connectivity check."""
    ok_key = bool(os.getenv("OPENAI_API_KEY"))
    out = gpt_complete("Reply with just: OK", temperature=0.0, max_tokens=3)
    # include last error message if any
    try:
        from rag_system import _LAST_LLM_ERROR  # type: ignore
        err = _LAST_LLM_ERROR
    except Exception:
        err = None
    return {"has_key": ok_key, "response": out or "", "ok": ok_key and bool(out and out.strip()), "error": err}


@app.get("/api/prompts")
def prompts():
    return {"prompts": QUICK_PROMPTS}


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    sid = get_or_create_session(session_id)
    s = _SESSIONS[sid]
    return {"session_id": sid, "chat": [m.model_dump() for m in s.chat], "profile": s.profile}


@app.delete("/api/history/{session_id}")
def clear_history(session_id: str):
    sid = get_or_create_session(session_id)
    _SESSIONS[sid] = SessionData()
    return {"ok": True, "session_id": sid}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = get_or_create_session(req.session_id)
    s = _SESSIONS[sid]

    rag = get_rag()

    safe_q = sanitize_user_text(req.message)
    s.chat.append(Message(role="user", content=safe_q))

    # Convert history to the simple structure expected by RAG
    hist = [
        {"role": m.role, "content": m.content, "ts": m.ts}
        for m in s.chat
    ]

    ans: Answer = rag.answer(safe_q, s.profile, hist)

    # Save assistant reply and suggestions
    s.chat.append(Message(role="assistant", content=ans.text))
    if ans.profile_updates:
        s.profile.update(ans.profile_updates)

    cards: List[Dict[str, Any]] = []
    if ans.cards_df is not None and not ans.cards_df.empty:
        # Normalize to a list of dicts with stable keys
        for _, r in ans.cards_df.iterrows():
            cards.append({
                "bank_name": r.get("bank_name", ""),
                "card_name": r.get("card_name", ""),
                "annual_fee": r.get("annual_fee", ""),
                "key_benefits": r.get("key_benefits", ""),
                "description": r.get("description", ""),
                "website": r.get("website", ""),
                "card_type": r.get("card_type", ""),
            })
    s.last_cards = cards

    return ChatResponse(
        session_id=sid,
        answer=ans.text,
        suggestions=(ans.suggestions or [])[:6],
        profile=s.profile,
        cards=cards,
    )


def _jsonl_encode(obj: Dict[str, Any]) -> bytes:
    try:
        import json as _json
        return (_json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    except Exception:
        return b"\n"


@app.post("/api/chat_stream")
def chat_stream(req: ChatRequest):
    sid = get_or_create_session(req.session_id)
    s = _SESSIONS[sid]

    rag = get_rag()

    safe_q = sanitize_user_text(req.message)
    s.chat.append(Message(role="user", content=safe_q))

    hist = [
        {"role": m.role, "content": m.content, "ts": m.ts}
        for m in s.chat
    ]

    ans: Answer = rag.answer(safe_q, s.profile, hist)

    # Save assistant reply and suggestions
    s.chat.append(Message(role="assistant", content=ans.text))
    if ans.profile_updates:
        s.profile.update(ans.profile_updates)

    cards: List[Dict[str, Any]] = []
    if ans.cards_df is not None and not ans.cards_df.empty:
        for _, r in ans.cards_df.iterrows():
            cards.append({
                "bank_name": r.get("bank_name", ""),
                "card_name": r.get("card_name", ""),
                "annual_fee": r.get("annual_fee", ""),
                "key_benefits": r.get("key_benefits", ""),
                "description": r.get("description", ""),
                "website": r.get("website", ""),
                "card_type": r.get("card_type", ""),
            })
    s.last_cards = cards

    def streamer():
        yield _jsonl_encode({"event": "start", "session_id": sid})
        text = ans.text or ""
        # simple chunking by words
        buf = []
        for token in text.split(" "):
            buf.append(token)
            if len(buf) >= 20:
                chunk = " ".join(buf) + " "
                yield _jsonl_encode({"event": "delta", "text": chunk})
                buf = []
        if buf:
            chunk = " ".join(buf)
            yield _jsonl_encode({"event": "delta", "text": chunk})
        # end meta with suggestions and cards
        yield _jsonl_encode({
            "event": "end",
            "session_id": sid,
            "suggestions": (ans.suggestions or [])[:6],
            "profile": s.profile,
            "cards": cards,
        })

    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    os.makedirs("uploads", exist_ok=True)
    dest_path = os.path.join("uploads", f"cards_{uuid.uuid4().hex[:8]}.csv")
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    # Rebuild RAG with new dataset (global for now)
    try:
        os.environ["CREDIT_CARD_DATA_PATH"] = dest_path
        get_rag(force_reindex=True, data_path=dest_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index dataset: {e}")

    return UploadResponse(ok=True, message="Dataset uploaded & indexed", path=dest_path)


# Serve built frontend (if present under /app/static)
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
static_dir = os.path.abspath(static_dir)
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


