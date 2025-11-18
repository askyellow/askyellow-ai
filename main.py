from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import os
import uvicorn
import requests
import unicodedata
import re

# =============================================================
# 0. KNOWLEDGE ENGINE IMPORTS
# =============================================================
from yellowmind.knowledge_engine import load_knowledge, match_question
from yellowmind.identity_origin import try_identity_origin_answer

# =============================================================
# 1. ENVIRONMENT & OPENAI CLIENT
# =============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is missing")

# Model voor Yellowmind (kan je in Render/Replit als env var zetten)
YELLOWMIND_MODEL = os.getenv("YELLOWMIND_MODEL", "gpt-4o-mini")

# URL naar je Strato search-endpoint (opbouwen eigen data)
SQL_SEARCH_URL = os.getenv("SQL_SEARCH_URL", "https://askyellow.nl/search_knowledge.php")

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================
# 2. FASTAPI APP & CORS
# =============================================================

app = FastAPI(title="YellowMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TIP: later strakker zetten op askyellow.nl / shop.askyellow.nl
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# 3. HELPERS: FILE LOADING & SYSTEM PROMPT
# =============================================================

def load_file(path: str) -> str:
    """
    Lees een tekstbestand in als string.
    Als het bestand niet bestaat: geef een lege string terug (liever stille fout dan crash).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return "\n" + f.read().strip() + "\n"
    except FileNotFoundError:
        # Eventueel loggen:
        print(f"⚠️ Yellowmind config file niet gevonden: {path}")
        return ""


def build_system_prompt() -> str:
    """
    Bouwt de volledige YellowMind system prompt door alle txt-blokken aan elkaar te plakken.
    Volgorde:
    1) Master prompt (prioriteit, stijl, voorbeelden, mode-kiezen)
    2) Core identity / mission / values / communicatie
    3) Parents-profielen + mix-logic
    4) Behaviour & safety
    5) Knowledge & product regels
    6) Tone & modes
    """

    base = "yellowmind/"  # Aanpassen als jullie map-structuur anders is
    system_prompt = ""

    # 1) SYSTEM / MASTER PROMPT
    system_prompt += load_file(base + "system/yellowmind_master_prompt_v2.txt")

    # 2) CORE
    system_prompt += load_file(base + "core/core_identity.txt")
    system_prompt += load_file(base + "core/mission.txt")
    system_prompt += load_file(base + "core/values.txt")
    system_prompt += load_file(base + "core/introduction_rules.txt")
    system_prompt += load_file(base + "core/communication_baseline.txt")

    # 3) PARENTS
    system_prompt += load_file(base + "parents/parent_profile_brigitte.txt")
    system_prompt += load_file(base + "parents/parent_profile_dennis.txt")
    system_prompt += load_file(base + "parents/parent_profile_yello.txt")
    system_prompt += load_file(base + "parents/parent_mix_logic.txt")

    # 4) BEHAVIOUR & SAFETY
    system_prompt += load_file(base + "behaviour/behaviour_rules.txt")
    system_prompt += load_file(base + "behaviour/boundaries_safety.txt")
    system_prompt += load_file(base + "behaviour/escalation_rules.txt")
    system_prompt += load_file(base + "behaviour/uncertainty_handling.txt")
    system_prompt += load_file(base + "behaviour/user_types.txt")

    # 5) KNOWLEDGE & PRODUCT RULES
    system_prompt += load_file(base + "knowledge/knowledge_sources.txt")
    system_prompt += load_file(base + "knowledge/askyellow_site_rules.txt")
    system_prompt += load_file(base + "knowledge/product_rules.txt")
    system_prompt += load_file(base + "knowledge/no_hallucination_rules.txt")
    system_prompt += load_file(base + "knowledge/limitations.txt")

    # 6) TONE & MODES
    system_prompt += load_file(base + "tone/tone_of_voice.txt")
    system_prompt += load_file(base + "tone/branding_mode.txt")
    system_prompt += load_file(base + "tone/empathy_mode.txt")
    system_prompt += load_file(base + "tone/tech_mode.txt")
    system_prompt += load_file(base + "tone/storytelling_mode.txt")
    system_prompt += load_file(base + "tone/concise_mode.txt")

    return system_prompt.strip()


SYSTEM_PROMPT = build_system_prompt()
KNOWLEDGE_ENTRIES = load_knowledge()  # uit knowledge_engine.py


# =============================================================
# 4. SQL KNOWLEDGE SEARCH (HYBRIDE: PHP + PYTHON)
# =============================================================

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def jaccard_score(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    inter = wa.intersection(wb)
    union = wa.union(wb)
    return len(inter) / len(union)


def contains_score(question: str, candidate_q: str) -> float:
    nq = normalize(question)
    nc = normalize(candidate_q)
    if nc in nq or nq in nc:
        return 1.0
    return 0.0


def compute_match_score(user_q: str, cand_q: str) -> int:
    j = jaccard_score(user_q, cand_q)        # overlap woorden
    c = contains_score(user_q, cand_q)       # één bevat de ander

    raw = 0.7 * j + 0.3 * c
    score = int(raw * 100)
    return max(0, min(score, 100))


def search_sql_knowledge(question: str):
    """
    Roept search_knowledge.php aan op Strato om candidates op te halen,
    vervolgens in Python scoreren en de beste teruggeven.
    Verwacht JSON-lijst met dicts: { id, question, answer, ... }.
    """
    try:
        resp = requests.post(SQL_SEARCH_URL, data={"q": question}, timeout=3)
        if resp.status_code != 200:
            print("⚠️ SQL search HTTP status:", resp.status_code)
            return None

        data = resp.json()
    except Exception as e:
        print("⚠️ SQL search error:", e)
        return None

    if not data:
        return None

    best = None
    best_score = 0

    for row in data:
        cand_q = row.get("question", "")
        cand_a = row.get("answer", "")

        score = compute_match_score(question, cand_q)

        if score > best_score:
            best_score = score
            best = {
                "id": row.get("id"),
                "question": cand_q,
                "answer": cand_a,
                "score": score,
            }

    if not best:
        return None

    print(f"🧠 Best SQL candidate score={best_score} for Q='{best['question']}'")
    return best


# =============================================================
# 5. TONE / MODE DETECTION
# =============================================================

def detect_hints(question: str) -> dict:
    """
    Super simpele auto-detectie van mode_hint / context_type / user_type_hint.
    Dit is puur heuristiek; later uit te breiden of te vervangen.
    """
    q = question.lower()

    mode_hint = None
    context_type = None
    user_type_hint = None

    # Emotioneel / onzeker
    if any(x in q for x in ["ik voel me", "mislukt", "huil", "overprikkeld", "overwhelmed", "ik weet niet meer"]):
        mode_hint = "empathy"
        user_type_hint = "emotioneel"

    # Tech
    if any(x in q for x in ["api", "error", "foutmelding", "script", "bug", "dns", "shopify", "liquid"]):
        mode_hint = "tech"
        context_type = "general"

    # Branding / socials / AskYellow
    if any(x in q for x in ["askyellow", "yellowmind", "branding", "logo", "stijl", "instagram", "insta", "tiktok", "facebook", "caption", "post", "reel", "short"]):
        if mode_hint is None:
            mode_hint = "branding"
        context_type = "askyellow_shop"

    # Als nog niks bepaald is → general / auto
    if mode_hint is None:
        mode_hint = "auto"
    if context_type is None:
        context_type = "general"

    return {
        "mode_hint": mode_hint,
        "context_type": context_type,
        "user_type_hint": user_type_hint,
    }


# =============================================================
# 6. HELPERS: OPENAI CALL
# =============================================================

def call_yellowmind_llm(
    question: str,
    language: str,
    kb_answer: str | None,
    sql_match: dict | None,
    hints: dict | None,
) -> str:
    """
    Stuurt de vraag + system prompt + eventuele KB-kennis (JSON + SQL)
    + backend hints naar het model. Laat YellowMind zelf het uiteindelijke
    antwoord formuleren in AskYellow-stijl.
    """
    messages = []

    # SYSTEM: volledige brein + regels
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # SYSTEM: ASKYELLOW_KNOWLEDGE (JSON KB + SQL KB)
    knowledge_parts = []

    if kb_answer:
        knowledge_parts.append(
            "STATIC_KB (JSON / FAQ / site-informatie):\n"
            + kb_answer.strip()
        )

    if sql_match:
        knowledge_parts.append(
            "SQL_KB (geleerde vraag/antwoord uit database):\n"
            f"Vraag: {sql_match.get('question', '').strip()}\n"
            f"Antwoord: {sql_match.get('answer', '').strip()}\n"
            f"Match-score: {sql_match.get('score')}"
        )

    if knowledge_parts:
        knowledge_block = "[ASKYELLOW_KNOWLEDGE]\n" + "\n\n".join(knowledge_parts)
        messages.append({"role": "system", "content": knowledge_block})

    # SYSTEM: BACKEND_HINTS (mode, context, user_type, taal)
    if hints is None:
        hints = {}

    # Taal altijd als hint meegeven
    if language:
        hints.setdefault("user_language", language)

    hint_lines = [f"{k}: {v}" for k, v in hints.items() if v]
    if hint_lines:
        hints_block = "[BACKEND_HINTS]\n" + "\n".join(f"- {line}" for line in hint_lines)
        messages.append({"role": "system", "content": hints_block})

    # USER: de eigenlijke vraag
    messages.append({"role": "user", "content": question})

    # OpenAI Responses API
    response = client.responses.create(
        model=YELLOWMIND_MODEL,
        input=messages,
    )

    # Nieuw Responses-format: output[0].content[0].text
    try:
        text = response.output[0].content[0].text
    except Exception:
        # Fallback, just in case
        text = str(response)

    return text.strip()


# =============================================================
# 7. ENDPOINTS
# =============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Yellowmind backend draait 🚀"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.head("/")
async def head_root():
    # Fix voor 405 HEAD spam in logs
    return Response(status_code=200)


@app.post("/ask")
async def ask_ai(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    language = (data.get("language") or "nl").lower()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Geen vraag ontvangen. Stuur een veld 'question' mee in de body."},
        )

    # 1️⃣ Identity / origin quick win (bijv. “Wie ben jij?”)
    #    We proberen eerst met (question, language); als jullie oudere versie heeft 1 arg,
    #    valt hij terug op alleen question.
    identity_answer = None
    try:
        identity_answer = try_identity_origin_answer(question, language)
    except TypeError:
        try:
            identity_answer = try_identity_origin_answer(question)
        except Exception as e:
            print("⚠️ identity-origin error:", e)
            identity_answer = None

    if identity_answer:
        return {
            "answer": identity_answer,
            "source": "identity_origin",
            "kb_used": False,
            "sql_used": False,
            "sql_score": None,
            "hints": {},
        }

    # 2️⃣ SQL KNOWLEDGE LAYER (alleen approved = 1, via PHP) → eigen data
    sql_match_raw = search_sql_knowledge(question)
    sql_match = None
    if sql_match_raw and sql_match_raw.get("score", 0) >= 60:
        sql_match = sql_match_raw
    # Onder de 60 → te twijfelachtig, dan niet gebruiken als kennisblok

    # 3️⃣ JSON KNOWLEDGE ENGINE
    try:
        kb_answer = match_question(question, KNOWLEDGE_ENTRIES)
    except Exception as e:
        print("⚠️ knowledge engine error:", e)
        kb_answer = None

    # 4️⃣ Hints (mode, context, user_type, taal)
    hints = detect_hints(question)

    # 5️⃣ Yellowmind LLM met volledig brein + KB + hints
    try:
        final_answer = call_yellowmind_llm(
            question=question,
            language=language,
            kb_answer=kb_answer,
            sql_match=sql_match,
            hints=hints,
        )
        source = "yellowmind_llm"
    except Exception as e:
        print("🔴 Yellowmind LLM ERROR:", e)
        final_answer = (
            "⚠️ Ik kan op dit moment geen live antwoord ophalen. "
            "Probeer het over een paar seconden opnieuw."
        )
        source = "error"

    return {
        "answer": final_answer,
        "source": source,
        "kb_used": bool(kb_answer),
        "sql_used": bool(sql_match),
        "sql_score": sql_match["score"] if sql_match else None,
        "hints": hints,
    }


# =============================================================
# 8. LOCAL DEV STARTER
# =============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
