from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import os

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
# 4. SQL LOGGING UITGESCHAKELD
# =============================================================
def log_to_sql(question: str, answer: str):
    """
    SQL logging is uitgeschakeld. Frontend gebruikt log.php op Strato.
    """
    print("ℹ️ SQL logging disabled (frontend handles logging via log.php)")

# =============================================================
# 5. HEALTH CHECK
# =============================================================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Yellowmind backend draait 🚀"
    }

# =============================================================
# 6. HOOFD ENDPOINT: /ask
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

    # ---------------------------------------------------------
    # 1. IDENTITY ORIGIN
    # ---------------------------------------------------------
    try:
        identity_answer = try_identity_origin_answer(question, language)
    except Exception as e:
        print("⚠️ identity-origin error:", e)
        identity_answer = None

    if identity_answer:
        final_answer = identity_answer
        source = "identity"

    else:
        # -----------------------------------------------------
        # 2. KNOWLEDGE ENGINE
        # -----------------------------------------------------
        try:
            kb_answer = match_question(question, KB)
        except Exception as e:
            print("⚠️ knowledge engine error:", e)
            kb_answer = None

        if kb_answer:
            final_answer = kb_answer
            source = "knowledge"

        else:
            # -------------------------------------------------
            # 3. OPENAI FALLBACK
            # -------------------------------------------------
            try:
                system_msg = (
                    "Je bent Yellowmind, de AI van AskYellow. "
                    "Je antwoordt eerlijk, duidelijk en vriendelijk. "
                    "Blijf in dezelfde taal als de vraag."
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

    # ---------------------------------------------------------
    # 4. LOGGING (UIT)
    # ---------------------------------------------------------
    log_to_sql(question, final_answer)

    # ---------------------------------------------------------
    # 5. TERUG NAAR FRONTEND
    # ---------------------------------------------------------
    return {
        "answer": final_answer,
        "source": source,
    }
