import re
import pandas as pd

# -------- intent detection --------
def detect_intent(q: str) -> str:
    ql = (q or "").strip().lower()
    # Smalltalk
    if any(w in ql for w in ["hi","hello","hey","namaste","good morning","good evening","yo","sup","thank","thanks","help"]):
        return "smalltalk"
    # Explicit compare
    if "compare" in ql or " vs " in ql or "versus" in ql:
        return "compare"
    # General banking Q&A (avoid triggering recommendations)
    banking_qa_triggers = [
        "what is ", "difference between", "how to ", "how does ", "meaning of ", "define ",
        "benefits of ", "pros and cons", "cons of ", "eligibility for ", "requirements for ",
        "credit score", "cibil score", "debit card", "upi", "net banking", "kcc", "loan", "emi",
    ]
    if any(w in ql for w in banking_qa_triggers):
        return "banking_qa"
    # Recommendation keywords
    card_terms = [
        "credit card","card","cashback","lounge","travel","airport","rewards","fuel",
        "groceries","shopping","dining","movies","no annual fee","annual fee","cibil",
        "eligibility","income","limit","premium","lakh","fee","bank"
    ]
    if any(w in ql for w in card_terms):
        return "recommend"
    return "unknown"

def required_profile_slots():
    return ["income", "cibil", "max_fee", "categories", "bank"]

def pretty_slot_names():
    return {
        "income": "Monthly income (₹)",
        "cibil": "CIBIL",
        "max_fee": "Max annual fee (₹)",
        "categories": "Spend categories",
        "bank": "Preferred bank"
    }

def route_query(query: str, vector_empty: bool) -> bool:
    q = (query or "").lower()
    if vector_empty:
        return True
    return any(w in q for w in [
        "latest","news","recent","2024","2025","2026","updated","policy","new rules","change","revised","launch"
    ])

def extract_filters_from_query(query: str):
    q = (query or "").lower()
    out = {"bank": None, "max_fee": None, "categories": None}
    m = re.search(r"\b(hdfc|sbi|icici|axis|kotak|rbl|yes|idfc|idbi|amex|indusind)\b", q)
    if m: out["bank"] = m.group(1).upper()
    m = re.search(r"(?:under|below|less than|upto|up to|<=)\s*₹?\s*([\d,]{3,6})", q)
    if m:
        try: out["max_fee"] = int(m.group(1).replace(",", ""))
        except: pass
    cats = []
    for w in ["cashback","travel","lounge","fuel","shopping","online","dining","movies","groceries","rewards","forex","international","priority pass","lounge access","airport"]:
        if w in q: cats.append("Lounge" if w=="lounge" else w.capitalize())
    if cats: out["categories"] = sorted(set(cats))
    return out

# --- comparison parsing (e.g., "compare A vs B") ---
def extract_compare_pair(query: str):
    q = (query or "")
    # split by connectors: vs, versus, compare, with, and
    parts = re.split(r"\b(?:vs|versus|compare|with|and)\b", q, flags=re.I)
    parts = [p.strip(" ,.-") for p in parts if p.strip()]
    if len(parts) >= 2:
        a, b = parts[-2], parts[-1]
        def clean(name):
            name = re.sub(r"\b(card|credit|credit card|the|and|with)\b", "", name, flags=re.I)
            return re.sub(r"\s+", " ", name).strip()
        a, b = clean(a), clean(b)
        if len(a) >= 2 and len(b) >= 2:
            return a, b
    return None, None

# Fallback fusion (used when ranking model disabled)
def fuse_answers(df: pd.DataFrame, web_text: str) -> str:
    parts = []
    if df is not None and not df.empty:
        parts.append("### 🟡 Top matches from your dataset:")
        for _, r in df.iterrows():
            name = r.get("card_name",""); bank = r.get("bank_name",""); ctype = r.get("card_type","")
            desc = r.get("description",""); kb = r.get("key_benefits",""); site = r.get("website","")
            line = f"- **{name}** — {bank} ({ctype})\n  - {desc}\n  - **Benefits:** {kb}"
            if site: line += f"\n  - [Apply/Details]({site})"
            parts.append(line)
    else:
        parts.append("_No strong matches found in your uploaded dataset._")
    if isinstance(web_text, str) and web_text.strip():
        parts.append("### 🌐 Web highlights:\n" + web_text.strip())
    return "\n\n".join(parts).strip()

SMALLTALK_REPLIES = {
    "hello": "Hey! 👋 How can I help you today?",
    "thanks": "Happy to help! Tell me your **income, CIBIL, max fee, bank** and **benefits** to get a shortlist.",
    "help": "Share your **income (₹/month)**, **CIBIL**, **max fee**, preferred **bank**, and **benefits** (cashback / travel lounge / fuel / shopping / dining / groceries).",
}
