import os
import re
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

from data_processor import CreditCardDataProcessor
from vector_store import CreditCardVectorStore, LocalRetriever
from query_router import (
    route_query, detect_intent, SMALLTALK_REPLIES,
    extract_filters_from_query, extract_compare_pair, required_profile_slots as rq_slots,
    pretty_slot_names, fuse_answers
)
from web_search import CreditCardWebSearch

# Optional direct OpenAI SDK fallback (when LangChain client isn't available)
try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI_SDK = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI_SDK = False

# OpenAI setup (direct SDK preferred)
_OPENAI_OK = bool(os.getenv("OPENAI_API_KEY"))
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_FALLBACK_MODELS = [
    _OPENAI_MODEL,
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
]
_LAST_LLM_ERROR: Optional[str] = None

@dataclass
class Answer:
    text: str
    cards_df: Optional[pd.DataFrame] = None
    profile_updates: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

def required_profile_slots() -> List[str]:
    return ["income", "cibil", "max_fee", "categories", "bank"]

def pretty_slot(k: str) -> str:
    return pretty_slot_names().get(k, k.capitalize())
def gpt_complete(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> Optional[str]:
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None
    # Single path: OpenAI SDK directly if key exists
    if _OPENAI_OK and _HAS_OPENAI_SDK:
        client = OpenAI()
        for model in _OPENAI_FALLBACK_MODELS:
            if not model:
                continue
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content  # type: ignore
            except Exception as e:
                _LAST_LLM_ERROR = f"{type(e).__name__}: {e} (model={model})"
                continue
    return None

# -------- profile parsing from natural text --------
def parse_profile_hints(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    out: Dict[str, Any] = {}
    m = re.search(r"(?:₹|rs\.?\s*)?([\d,]{4,})\s*(?:/m|per month|monthly)?", t)
    if m:
        try: out["income"] = int(m.group(1).replace(",", ""))
        except: pass
    m = re.search(r"\b([3-8]\d{2}|900)\b", t)
    if m:
        v = int(m.group(1))
        if 300 <= v <= 900: out["cibil"] = v
    m = re.search(r"(?:max|under|below|<=|less than|upto|up to)\D*([\d,]{3,6})\s*(?:fee|annual fee)?", t)
    if m:
        try: out["max_fee"] = int(m.group(1).replace(",", ""))
        except: pass
    cats_map = {
        "cashback": "Cashback", "travel": "Travel", "lounge": "Lounge", "fuel": "Fuel",
        "shopping": "Shopping", "online": "Online", "dining": "Dining", "movies": "Movies",
        "groceries": "Groceries", "rewards": "Rewards"
    }
    cats = [v for k, v in cats_map.items() if k in t]
    if cats: out["categories"] = sorted(set(cats))
    m = re.search(r"\b(hdfc|sbi|icici|axis|kotak|rbl|yes|idfc|idbi|amex|indusind)\b", t)
    if m: out["bank"] = m.group(1).upper()
    return out

RECOMMEND_PROMPT = """You are a senior Indian banking product specialist.
Task: From candidate cards, select the BEST up to 3 for the user and justify succinctly.

PROFILE (JSON):
{profile}

USER QUERY:
{query}

CANDIDATES (JSON Lines):
{candidates}

Hard constraints:
- If profile.bank is set, recommend only that bank's cards when available; if none, state and broaden.
- Respect profile.max_fee when present.
- Prefer cards aligned to profile.categories when present.

Output strictly in compact Markdown:
1. One-line need summary
2. Ranked list (1–3): **Card Name — Bank (Type)**, 1 short reason, 3–5 bullets of concrete perks, annual fee (₹), website link
3. If bank constraint had no matches, one sentence about fallback
Note: Use ONLY facts from the candidates. Do NOT add imaginary benefits.
"""

COMPARE_PROMPT = """You are a banking product specialist.
Compare the two cards and advise which profile suits each.

USER QUERY:
{query}

CARD A (JSON): {card_a}
CARD B (JSON): {card_b}

Write Markdown:
- Short overview (1–2 lines)
- Table: Annual fee, Reward type, Lounge/Fuel/Online/Dining, Milestone/Waiver, Foreign markup (if available)
- Bullet guidance: who should pick A vs B (income/CIBIL/spend)
- One-line verdict
Only use facts present in the inputs.
"""

def jsonl_from_df(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.head(8).iterrows():
        rows.append(json.dumps({
            "bank_name": r.get("bank_name",""),
            "card_name": r.get("card_name",""),
            "annual_fee": r.get("annual_fee",""),
            "key_benefits": r.get("key_benefits",""),
            "description": r.get("description",""),
            "website": r.get("website",""),
            "card_type": r.get("card_type",""),
        }, ensure_ascii=False))
    return "\n".join(rows)

def row_json(row: pd.Series) -> str:
    return json.dumps({
        "bank_name": row.get("bank_name",""),
        "card_name": row.get("card_name",""),
        "annual_fee": row.get("annual_fee",""),
        "key_benefits": row.get("key_benefits",""),
        "description": row.get("description",""),
        "website": row.get("website",""),
        "card_type": row.get("card_type",""),
    }, ensure_ascii=False)

def slot_chips(slot: str) -> List[str]:
    return {
        "income": ["₹25,000 / month", "₹60,000 / month", "₹1,00,000 / month"],
        "cibil": ["CIBIL 650", "CIBIL 720", "CIBIL 800"],
        "max_fee": ["Under ₹500", "Under ₹1000", "Under ₹3000"],
        "categories": ["cashback", "travel lounge", "fuel", "shopping online", "dining", "groceries"],
        "bank": ["SBI", "HDFC", "ICICI", "Axis", "Kotak"],
    }.get(slot, [])

class CreditCardRAG:
    def __init__(self, force_reindex: bool=False, data_path: Optional[str] = None):
        warnings.filterwarnings("ignore", category=UserWarning)
        path = data_path or os.getenv("CREDIT_CARD_DATA_PATH", "credit_cards_dataset.csv")
        self.data = CreditCardDataProcessor(path)
        self.vs = CreditCardVectorStore(force_reindex=force_reindex, data_path=path)
        self.retriever: LocalRetriever = self.vs.as_retriever(k=10)
        self.web = CreditCardWebSearch()

    # ---- smalltalk
    def _smalltalk(self, q: str) -> Answer:
        ql = (q or "").lower()
        if "thank" in ql:
            return Answer(text=SMALLTALK_REPLIES["thanks"], suggestions=["Recommend a card", "Compare cards"])
        if "help" in ql:
            return Answer(text=SMALLTALK_REPLIES["help"], suggestions=["cashback", "travel lounge", "fuel"])
        out = gpt_complete(
            "You are a warm, concise banking assistant. Reply in ≤2 lines and offer help with credit card advice.\n"
            f"User: {q}"
        )
        if isinstance(out, str) and out.strip():
            return Answer(text=out, suggestions=["cashback", "travel lounge", "fuel"])
        return Answer(text=SMALLTALK_REPLIES.get("hello", "Hello! How can I help you today?"), suggestions=["cashback", "shopping online", "fuel"])

    # ---- general banking Q&A (e.g., difference between credit and debit card)
    def _banking_qa(self, query: str) -> Answer:
        resp = gpt_complete(
            "You are a concise Indian banking expert.\n"
            "Task: If the user asks 'what is X', define X clearly in 2–4 lines (not differences).\n"
            "Use bold for key terms; add 2 bullets for key uses/features if relevant.\n"
            "If the user asks for advice later, prompt for income, CIBIL, max fee, bank, categories.\n"
            f"Question: {query}"
        )
        if isinstance(resp, str) and resp.strip():
            return Answer(text=resp, suggestions=["Recommend a credit card", "Compare two cards"])
        # Fallback: simple definition, not comparison
        return Answer(text=(
            "A debit card is a payment card linked to your bank account. When you pay or withdraw cash,\n"
            "the amount is deducted immediately from your available balance.\n\n"
            "- Works at ATMs, POS, and online.\n"
            "- No credit line or interest; usually lower fees than credit cards."
        ), suggestions=["Recommend a credit card", "Compare two cards"])

    # ---- recommend ranking (unified: dataset + optional web)
    def _llm_rank(self, query: str, profile: Dict[str, Any], df: pd.DataFrame, web_text: str = "") -> str:
        if (df is None or df.empty) and not web_text:
            return "_No strong matches found in your dataset or on the web._"
        base = RECOMMEND_PROMPT
        if isinstance(web_text, str) and web_text.strip():
            base += "\nWEB FINDINGS (bullets):\n" + web_text.strip() + "\n"
            base += "\nIncorporate corroborated web facts (add source in parentheses), but avoid contradictions.\n"
        prompt = base.format(
            profile=json.dumps({
                "income": profile.get("income"),
                "cibil": profile.get("cibil"),
                "max_fee": profile.get("max_fee"),
                "categories": profile.get("categories"),
                "bank": profile.get("bank"),
            }, ensure_ascii=False),
            query=query,
            candidates=jsonl_from_df(df if df is not None else pd.DataFrame())
        )
        out = gpt_complete(prompt, temperature=0.2, max_tokens=900)
        if isinstance(out, str) and out.strip():
            return out
        # Fallback: merge simple dataset summary with web highlights
        return fuse_answers(df if df is not None else pd.DataFrame(), web_text or "")

    # ---- compare two specific cards (no slot-filling)
    def _compare(self, query: str, name_a: str, name_b: str) -> Answer:
        def find_one(name: str) -> Optional[pd.Series]:
            df = self.retriever.search(name, k=5)
            if df is None or df.empty: return None
            mask = df["card_name"].str.lower().str.contains(name.lower(), na=False)
            return df[mask].iloc[0] if mask.any() else df.iloc[0]

        row_a = find_one(name_a) if name_a else None
        row_b = find_one(name_b) if name_b else None
        if row_a is None or row_b is None:
            # Try web search summary as fallback
            web = self.web.search_credit_card(f"Compare {name_a} vs {name_b} India credit card fees benefits")
            if web:
                return Answer(text=("I couldn’t find both cards locally.\n\n### 🌐 Web highlights\n" + web))
            return Answer(text="I couldn’t find both cards in your dataset. Check spellings or upload a richer CSV.")

        prompt = COMPARE_PROMPT.format(
            query=query,
            card_a=row_json(row_a),
            card_b=row_json(row_b),
        )
        text = gpt_complete(prompt, temperature=0.2, max_tokens=800)
        if not isinstance(text, str) or not text.strip():
            text = self._compare_fallback(row_a, row_b)

        return Answer(text=text, cards_df=pd.DataFrame([row_a, row_b]))

    def _compare_fallback(self, a: pd.Series, b: pd.Series) -> str:
        def fmt(r: pd.Series) -> str:
            return (
                f"**{r.get('card_name','')}** — {r.get('bank_name','')} ({r.get('card_type','')})\n"
                f"- Annual fee: ₹{r.get('annual_fee','')}\n"
                f"- Perks: {r.get('key_benefits','')}\n"
                f"- Notes: {r.get('description','')}\n"
                f"- Link: {r.get('website','')}\n"
            )
        return "### 🔍 Side-by-side\n\n" + fmt(a) + "\n" + fmt(b) + \
               "\n> Tip: Tell me your main spend (fuel, travel+lounge, online shopping, dining, groceries) and fee comfort; I’ll suggest which fits you better."

    # ---- main
    def answer(self, query: str, profile: Dict[str, Any], history: List[Dict[str, Any]]) -> Answer:
        intent = detect_intent(query)

        # smalltalk
        if intent == "smalltalk":
            return self._smalltalk(query)

        # general banking Q&A
        if intent == "banking_qa":
            return self._banking_qa(query)

        # comparison path
        if intent == "compare":
            a, b = extract_compare_pair(query)
            if a and b:
                return self._compare(query, a, b)

        # enrich profile from query
        updates = parse_profile_hints(query)
        qf = extract_filters_from_query(query)
        updates.update({k: v for k, v in qf.items() if v is not None})
        if updates: profile.update(updates)

        # slot-ask only for recommendation
        if intent == "recommend":
            missing = [k for k in required_profile_slots() if not profile.get(k)]
            if missing:
                k = missing[0]
                ask = {
                    "income": "What’s your **monthly income (₹)**?",
                    "cibil": "What’s your **CIBIL score** (300–900)?",
                    "max_fee": "What’s the **max annual fee (₹)** you’re okay with?",
                    "categories": "Which **benefits** matter? (cashback / travel lounge / fuel / shopping online / dining / groceries)",
                    "bank": "Any preferred **bank** (SBI/HDFC/ICICI/Axis/Kotak)?",
                }
                return Answer(text=ask[k], profile_updates=updates, suggestions=slot_chips(k))

        # retrieve with bank/fee/category constraints
        cards_df = self.retriever.search(
            query=query,
            bank=profile.get("bank") or qf.get("bank"),
            max_fee=profile.get("max_fee") or qf.get("max_fee"),
            categories=profile.get("categories") or qf.get("categories"),
        )

        # Optional web
        need_web = route_query(query, cards_df is None or cards_df.empty)
        web = self.web.search_credit_card(query) if need_web else ""

        # Unified answer (dataset + web)
        explanation = self._llm_rank(query, profile, cards_df, web)

        sugg = ["Compare two cards", "Show low annual-fee cards", "Fuel benefits", "Cashback options"]
        return Answer(text=explanation, cards_df=cards_df, profile_updates=updates, suggestions=sugg)

    # legacy helper
    def answer_query(self, query: str, return_df: bool=False):
        ans = self.answer(query, {k: None for k in required_profile_slots()}, [])
        return (ans.text, ans.cards_df) if return_df else ans.text
