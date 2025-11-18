from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import os
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

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is missing")

client = OpenAI(api_key=api_key)

# URL naar je Strato search-endpoint
SQL_SEARCH_URL = os.getenv("SQL_SEARCH_URL", "https://askyellow.nl/search_knowledge.php")

# =============================================================
# 2. FASTAPI SETUP
# =============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # later strakker instellen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# 3. LOAD KNOWLEDGE ENGINE (bij startup)
# =============================================================
KB = load_knowledge()

# =============================================================
# 4. HULPFUNCTIES VOOR NORMALISATIE & SCORE
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

    # simpele weging
    raw = 0.7 * j + 0.3 * c
    score = int(raw * 100)
    return max(0, min(score, 100))


# =============================================================
# 5. SQL KNOWLEDGE SEARCH (HYBRIDE: PHP + PYTHON)
# =============================================================
def search_sql_knowledge(question: str):
    """
    Roept search_knowledge.php aan op Strato om candidates op te halen,
    vervolgens in Python scoreren en de beste teruggeven.
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


def rewrite_answer_with_gpt(user_question: str, base_answer: str, language: str, mode: str) -> str:
    """
    mode: 'direct', 'rewrite', 'hybrid'
    """
    if mode == "direct":
        return base_answer

    if language not in ["nl", "en"]:
        language = "nl"

    if mode == "rewrite":
        user_instruction = (
            f"Herschrijf het volgende antwoord in goed leesbaar {language.upper()}, "
            "vriendelijk en helder, in AskYellow-stijl. Verander geen feiten.\n\n"
            f"Antwoord:\n{base_answer}"
        )
    else:  # hybrid
        user_instruction = (
            f"De gebruiker vraagt: '{user_question}'.\n\n"
            f"Er is eerder dit antwoord gegeven:\n{base_answer}\n\n"
            "Gebruik dit als basis, maar verbeter, structureer en vul aan waar nodig. "
            f"Antwoord in {language.upper()} en in de stijl van een behulpzame AskYellow assistent."
        )

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Je bent Yellowmind, de AI van AskYellow. "
                        "Je antwoordt eerlijk, duidelijk en vriendelijk. "
                        "Blijf in dezelfde taal als de gebruiker."
                    ),
                },
                {"role": "user", "content": user_instruction},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("⚠️ GPT rewrite error:", e)
        return base_answer


# =============================================================
# 6. SQL LOGGING UITGESCHAKELD (FRONTEND DOET DIT VIA log.php)
# =============================================================
def log_to_sql(question: str, answer: str):
    print("ℹ️ SQL logging disabled in backend (frontend gebruikt log.php op Strato).")


# =============================================================
# 7. HEALTH CHECK
# =============================================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Yellowmind backend draait 🚀"}

# =============================================================
# 8. HOOFD ENDPOINT: /ask
# =============================================================
@app.post("/ask")
async def ask_ai(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    language = (data.get("language") or "nl").lower()

    if not question:
        return {
            "answer": "Ik heb geen vraag ontvangen. Typ eerst een vraag in het veld.",
            "source": "system",
        }

    final_answer = None
    source = "unknown"

    # 1️⃣ IDENTITY ORIGIN
    try:
        identity_answer = try_identity_origin_answer(question, language)
    except Exception as e:
        print("⚠️ identity-origin error:", e)
        identity_answer = None

    if identity_answer:
        final_answer = identity_answer
        source = "identity"

    else:
        # 2️⃣ SQL KNOWLEDGE LAYER (alleen approved = 1)
        sql_match = search_sql_knowledge(question)

        if sql_match and sql_match["score"] >= 60:
            score = sql_match["score"]
            base_answer = sql_match["answer"]

            if score >= 95:
                mode = "direct"   # 1-op-1 overnemen
            elif score >= 80:
                mode = "rewrite"  # zelfde inhoud, mooier verwoord
            else:
                mode = "hybrid"   # basis + verrijking

            final_answer = rewrite_answer_with_gpt(question, base_answer, language, mode)
            source = f"sql_{mode}"

        else:
            # 3️⃣ JSON KNOWLEDGE ENGINE
            try:
                kb_answer = match_question(question, KB)
            except Exception as e:
                print("⚠️ knowledge engine error:", e)
                kb_answer = None

            if kb_answer:
                final_answer = kb_answer
                source = "knowledge"
            else:
                # 4️⃣ OPENAI FALLBACK
                try:
                    system_msg = (
                        "Je bent Yellowmind, de AI van AskYellow. "
                        "Je antwoordt eerlijk, duidelijk en vriendelijk. "
                        "Blijf in dezelfde taal als de vraag. "
                        "Als een vraag dubbelzinnig is, ga uit van de meest neutrale, "
                        "informele betekenis. Bij korte vragen als 'Doe je het?' of "
                        "'Ben je wakker?' geef je een vriendelijke bevestiging dat je actief bent."
                    )

                    completion = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": question},
                        ],
                    )

                    final_answer = completion.choices[0].message.content
                    source = "openai"
                except Exception as e:
                    print("🔴 OpenAI ERROR:", e)
                    final_answer = (
                        "⚠️ Ik kan op dit moment geen live antwoord ophalen. "
                        "Probeer het over een paar seconden opnieuw."
                    )
                    source = "error"

    # 5️⃣ LOGGING (nog steeds via frontend)
    log_to_sql(question, final_answer)

    return {
        "answer": final_answer,
        "source": source,
    }
