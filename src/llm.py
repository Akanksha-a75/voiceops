"""Thin Groq wrapper: retries on rate limits, optional JSON mode."""
import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", MODEL)

_client = None


def client():
    global _client
    if _client is None:
        key = os.getenv("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_API_KEY is empty. Put your Groq key in .env")
        _client = Groq(api_key=key)
    return _client


def chat(messages, json_mode=False, temperature=0.0, tools=None,
         max_tokens=2500, model=None):
    kwargs = dict(model=model or MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if "gpt-oss" in (model or MODEL):
        kwargs["reasoning_effort"] = "low"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    last = None
    for attempt in range(7):
        try:
            return client().chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            if any(s in msg for s in ("429", "rate limit", "rate_limit", "overloaded", "503", "timeout")):
                time.sleep(min(2 ** attempt * 2, 40))
                continue
            raise
    raise RuntimeError(f"LLM failed after retries: {last}")


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {"decision": "answer", "reply": (text or "").strip()}


def chat_json(messages, **kw):
    r = chat(messages, json_mode=True, **kw)
    return parse_json(r.choices[0].message.content)


def chat_text(messages, **kw):
    r = chat(messages, **kw)
    return (r.choices[0].message.content or "").strip()