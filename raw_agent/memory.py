"""
raw_agent/memory.py — Long-term memory (process band hone ke baad bhi yaad)
===========================================================================

KYUN BANI:
    Step 4a ki short-term memory (self.messages) sirf RAM mein hai. Program band
    hote hi sab bhool jaata hai. Yahan facts ek JSON FILE mein save hote hain,
    taaki agli baar chat kholne pe bhi agent ko yaad rahe.

KYA KARTI HAI:
    remember(fact) -> fact file mein save (duplicate aur sensitive data reject)
    recall(query)  -> query se milte-julte facts dhoondh ke lautata hai

SHORT-TERM vs LONG-TERM (interview):
    short-term = messages list, RAM mein, EK conversation tak
    long-term  = file/DB, conversations aur restarts ke paar
    LangGraph mein: checkpointer = short-term (per thread_id),
                    Store (InMemoryStore / Postgres) = long-term (namespaces)

GUARDRAIL CODE MEIN (prompt mein nahi):
    Card/ID jaise lambe numbers aur password/OTP words wale facts REJECT hote hain.
    Model bhool bhi jaaye, tab bhi save nahi honge. (Step 3e principle:
    jo code enforce kar sakta hai, use prompt pe mat chhodo.)

LIMITATION (jaan-boojh ke):
    recall abhi KEYWORD overlap se dhoondhta hai. "Profession" poochha aur fact
    mein "engineer" likha ho, toh match nahi hoga. Isi gap ko Phase 3 ke
    embeddings / semantic search bharenge.
"""

import json
import os
import re
import uuid
from datetime import datetime

# File ka path. Module-level variable rakha hai, taaki evals ise badal ke
# ALAG file use kar sakein (asli memory gandi na ho).
MEMORY_FILE = os.getenv("MEMORY_FILE", "data/memory.json")

# Sensitive patterns: 12+ digit numbers (card/Aadhaar jaise) ya secret words.
_SENSITIVE = [
    re.compile(r"\d[\d\s-]{10,}\d"),
    re.compile(r"(?i)\b(password|passwd|pin|otp|cvv)\b"),
]

# Recall mein in common words ko ignore karo, warna har fact har query se "match" karega.
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "what", "my", "i", "me", "user", "users",
              "s", "of", "to", "do", "you", "know", "about", "and", "in", "for", "their"}


def _load() -> list:
    """File se saare facts padho. File nahi hai toh khaali list."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(items: list):
    """Saare facts file mein likho (folder na ho toh bana do)."""
    os.makedirs(os.path.dirname(MEMORY_FILE) or ".", exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def _words(text: str) -> set:
    """Text ko chhote-letter words ke set mein todo, stopwords hata ke."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def remember(fact: str) -> str:
    """Ek fact long-term memory mein save karo."""
    fact = (fact or "").strip()
    if not fact:
        return "Error: fact khaali hai"
    for pattern in _SENSITIVE:                         # GUARDRAIL: code-level, model pe nirbhar nahi
        if pattern.search(fact):
            return "Refused: sensitive info (card/ID number ya password jaisa) memory mein save nahi hoti"
    items = _load()
    if any(i["fact"].lower() == fact.lower() for i in items):
        return "Already remembered"
    items.append({
        "id": uuid.uuid4().hex[:8],
        "fact": fact,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    _save(items)
    return f"Saved: {fact}"


def recall(query: str = "") -> str:
    """Query ke words se milte facts dhoondho (keyword overlap). Top 5 lautao."""
    items = _load()
    if not items:
        return "No memories saved yet"
    q = _words(query)
    scored = sorted(((len(q & _words(i["fact"])), i) for i in items),
                    key=lambda x: x[0], reverse=True)
    hits = [i["fact"] for score, i in scored if score > 0][:5]
    if not hits:
        # Keyword match nahi mila: haal ke 5 facts de do, taaki model khud dekh le.
        # (Yeh semantic search ki kami ka jugaad hai. Phase 3 mein sudhrega.)
        hits = [i["fact"] for i in items[-5:]]
    return json.dumps(hits, ensure_ascii=False)


def all_memories() -> list:
    """Chat ke /memory command ke liye: saare facts (system state se verify karne ko)."""
    return _load()