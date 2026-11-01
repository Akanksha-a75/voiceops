"""Single-question answering.
Step 1: prompt-only baseline (no documents).
Step 2: RAG - retrieve KB chunks + catalog rows and answer only from them.
"""
import re

from .kb import KB, format_context
from .llm import REWRITE_MODEL, chat_json, chat_text

_kb = None


def get_kb():
    global _kb
    if _kb is None:
        _kb = KB()
    return _kb


DECISIONS = """"decision" means:
- answer: you answered, or you told the customer the documents do not cover it
- clarify: a needed detail is missing, so you ask ONE short question
- escalate: a human colleague must decide (see escalation rules)
- refuse: the request breaks policy, so you decline politely and say why"""

JSON_RULE = 'Reply ONLY with a JSON object: {"decision": "answer|clarify|escalate|refuse", "reply": "<what you say to the customer>"}'

STEP1_SYSTEM = f"""You are ZentroMart's AI support assistant. ZentroMart is an Indian online store selling groceries and electronics. You help customers with deliveries, returns, refunds, cancellations and warranty.
- Start by telling the customer you are ZentroMart's AI support assistant.
- Reply in the customer's language (English, Hindi in Devanagari, or Hinglish in Latin script).
- Keep the reply to two or three short sentences.
- If you are not sure of a fact, say you do not have that information and offer to connect a human colleague.
{DECISIONS}
{JSON_RULE}"""

STEP2_SYSTEM = f"""You are ZentroMart's AI support assistant. ZentroMart is an Indian online store selling groceries and electronics.
Today's date is 20 October 2026.

Answer ONLY from the CONTEXT given with each question (company documents and catalog rows).
- Start by telling the customer you are ZentroMart's AI support assistant.
- Reply in the customer's language (English, Hindi in Devanagari, or Hinglish in Latin script).
- Keep the reply to two or three short sentences. Write dates with the month name and year, e.g. 27 October 2026.
- If documents disagree, follow this order: current policy, agent guide, help article, FAQ, marketing page. Never use superseded documents.
- Business days are Monday to Saturday, excluding 2026-08-15, 2026-10-02, 2026-10-19 and 2026-11-09. Day 1 is the first business day AFTER the start date. Return windows are calendar days counted from the delivery date.
- If the CONTEXT does not cover the question, say you do not have information on that and offer to connect a human colleague. Never invent a policy, date, amount or fee.
- If a needed detail is missing (for example which product), ask ONE short clarifying question.
- Never promise compensation, refunds or exceptions that a policy does not allow.
{DECISIONS}
{JSON_RULE}"""

VALID = {"answer", "clarify", "escalate", "refuse"}


def _decision(d):
    d = str(d or "answer").strip().lower()
    return d if d in VALID else "answer"


def baseline_answer(question):
    out = chat_json([{"role": "system", "content": STEP1_SYSTEM},
                     {"role": "user", "content": question}])
    return {"decision": _decision(out.get("decision")),
            "reply": str(out.get("reply", "")), "sources": []}


REWRITE_PROMPT = (
    "Rewrite the customer's message as a short English search query (max 15 words) for a "
    "support knowledge base. Use these words where they fit: return, refund, cancellation, "
    "delivery, late, damaged item, warranty, Assured, coupon, COD, wallet, reverse pickup, "
    "Plus membership, business days, human colleague. Keep product names, SKUs and numbers. "
    "Output only the query."
)
_warned = False


def search_query(question, use_rewrite=True):
    """Raw question + an English rewrite (helps Hindi / Hinglish / slang)."""
    global _warned
    if not use_rewrite:
        return question
    try:
        rw = chat_text([{"role": "system", "content": REWRITE_PROMPT},
                        {"role": "user", "content": question}],
                       max_tokens=60, model=REWRITE_MODEL)
        return f"{question} {rw}"
    except Exception as e:  # noqa: BLE001
        if not _warned:
            print(f"[warn] query rewrite failed ({e}); using raw question")
            _warned = True
        return question


def rag_answer(question, k=5, use_rewrite=True):
    kb = get_kb()
    q = search_query(question, use_rewrite)
    chunks = kb.search(q, k=k)
    prods = kb.products_for(question)
    ctx = format_context(chunks, prods)
    out = chat_json([
        {"role": "system", "content": STEP2_SYSTEM},
        {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nCUSTOMER QUESTION:\n{question}"},
    ])
    sources = [c["id"] for c in chunks] + [f'products.csv:{p["sku"]}' for p in prods]
    return {"decision": _decision(out.get("decision")),
            "reply": str(out.get("reply", "")), "sources": sources}