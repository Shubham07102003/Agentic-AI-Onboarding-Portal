import re

_PATS = [
    # Indian phone numbers (basic)
    re.compile(r"\b(?:\+?91[-\s]?)?[6-9]\d{9}\b"),
    # PAN (AAAAA9999A)
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    # Email
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Aadhaar (12 digits; very rough)
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
]

def sanitize_user_text(text: str) -> str:
    if not text: return text
    out = text
    for pat in _PATS:
        out = pat.sub("[redacted]", out)
    return out
