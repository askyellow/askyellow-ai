# search_v2/query_builder.py

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a Dutch e-commerce search decision engine.

Your task is to read the conversation and decide whether:
1. the system should ask one short follow-up question, or
2. the system is ready to search, or
3. the system should first give advice instead of searching.

Return ONLY valid JSON with exactly these fields:
- proposed_query (string or null)
- is_ready_to_search (boolean)
- confidence (number between 0 and 1)
- clarification_question (string or null)
- response_mode (string: "advice" or "search")

Rules:
- Output JSON only. No explanations.
- Use Dutch for proposed_query and clarification_question.
- Be concise and practical.
- Never return extra keys.

Decision logic:
- If the user clearly wants to shop or compare products now, use response_mode = "search".
- If the user is still asking for guidance about what kind of product they need, use response_mode = "advice".
- If one short follow-up question would significantly improve search quality, set is_ready_to_search = false.
- If the request is specific enough to search well, set is_ready_to_search = true.

Concrete product rule:
- If the user explicitly mentions a concrete product noun such as tv, televisie, fiets, magnetron, boormachine, stofzuiger, fatbike, or similar, treat the product as already known.
- Do NOT ask which product within the category they mean.
- Instead, ask about preferences, constraints, or features such as size, usage, type, budget, or important specifications.

Follow-up question rules:
- Ask at most ONE short Dutch clarification question.
- Only ask a question if it will clearly improve the results.
- Prefer asking about one high-impact attribute such as size, use case, type, or budget.
- Never ask the user to repeat the product if the product is already clear.
- Never ask a broad category question like "Wat zoek je binnen beeld en geluid?" when the product is already known.
- If the latest user input is negative or dismissive, do not invent a strange new question. Either continue logically or be ready to search if enough is known.

Search query rules:
- proposed_query must be null when is_ready_to_search = false.
- clarification_question must be null when is_ready_to_search = true.
- If is_ready_to_search = true, proposed_query must be a short, clear Dutch shopping query using the known product and most relevant constraints from the conversation.
- Do not make the query unnecessarily long.
- Include useful constraints such as budget, size, intended use, technology, or target user when known.
- Do not invent constraints.

Advice rules:
- Use response_mode = "advice" only when the user is still deciding what type of product they need.
- If the user already knows the product and wants options, prefer response_mode = "search".

Confidence:
- Use a number between 0 and 1.
- Use higher confidence when product + intent are clear.
- Use lower confidence when the conversation is ambiguous.

Contract:
- If is_ready_to_search = true:
  - proposed_query must be a non-empty string
  - clarification_question must be null
- If is_ready_to_search = false:
  - proposed_query must be null
  - clarification_question must be a non-empty string
""".strip()


# ----------------------------
# Helpers
# ----------------------------

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON even if model adds extra text (we disallow it, but be resilient).
    """
    text = _strip_code_fences(text)

    # If the model accidentally adds leading/trailing junk, try to extract the first JSON object.
    if not text.startswith("{"):
        start = text.find("{")
        if start != -1:
            text = text[start:]
    if not text.endswith("}"):
        end = text.rfind("}")
        if end != -1:
            text = text[: end + 1]

    return json.loads(text)


def _normalize_decision(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate required keys, types, and contract invariants.
    Raises ValueError if not valid.
    """
    required = {"proposed_query", "is_ready_to_search", "confidence", "clarification_question", "response_mode"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    
    if d["response_mode"] not in ["advice", "search"]:
        raise ValueError("response_mode must be 'advice' or 'search'")

    is_ready = d["is_ready_to_search"]
    if not isinstance(is_ready, bool):
        raise ValueError("is_ready_to_search must be boolean")

    conf = d["confidence"]
    if not isinstance(conf, (int, float)):
        raise ValueError("confidence must be a number")
    conf = float(conf)
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    pq = d["proposed_query"]
    cq = d["clarification_question"]

    if pq is not None and not isinstance(pq, str):
        raise ValueError("proposed_query must be string or null")
    if cq is not None and not isinstance(cq, str):
        raise ValueError("clarification_question must be string or null")

    # Contract: if ready => pq filled, cq null
    if is_ready:
        if not pq or not pq.strip():
            raise ValueError("is_ready_to_search=true but proposed_query is empty")
        if cq is not None and cq.strip():
            raise ValueError("is_ready_to_search=true but clarification_question is not null/empty")
        pq = pq.strip()
        cq = None
    else:
        if not cq or not cq.strip():
            raise ValueError("is_ready_to_search=false but clarification_question is empty")
        if pq is not None and pq.strip():
            raise ValueError("is_ready_to_search=false but proposed_query is not null/empty")
        cq = cq.strip()
        pq = None

    return {
        "proposed_query": pq,
        "is_ready_to_search": is_ready,
        "confidence": conf,
        "clarification_question": cq,
        "response_mode": d["response_mode"]

    }


def _conversation_to_text(conversation_history):
    lines = []
    for msg in conversation_history[-12:]:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

# ----------------------------
# Main function
# ----------------------------

def ai_build_search_decision(
    conversation_history: List[Dict[str, str]],
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Returns dict:
      - proposed_query (str|None)
      - is_ready_to_search (bool)
      - confidence (float 0..1)
      - clarification_question (str|None)

    Robust to minor formatting errors; retries with stricter instruction.
    """
    transcript = _conversation_to_text(conversation_history)

    user_prompt = f"""
Conversatie:
{transcript}
""".strip()

    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        extra = ""
        if attempt > 0:
            extra = f"""
Let op: Je vorige antwoord was ongeldig ({last_err}).
Geef nu ALLEEN geldig JSON dat exact voldoet aan het schema en de regels. Geen extra tekst.
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + ("\n\n" + extra if extra else "")},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        raw = (resp.choices[0].message.content or "").strip()

        try:
            parsed = _safe_json_loads(raw)
            normalized = _normalize_decision(parsed)
            return normalized
        except Exception as e:
            last_err = str(e)
            continue

    # Hard fallback: ask a generic but non-dumb clarify question (still not hardcoded per category)
    return {
        "proposed_query": None,
        "is_ready_to_search": False,
        "confidence": 0.0,
        "clarification_question": "Kun je één detail toevoegen zodat ik zeker weet welk type je bedoelt?",
        "response_mode": "search"

    }