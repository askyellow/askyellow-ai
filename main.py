from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import mysql.connector
import os

# =============================================================
# 0. KNOWLEDGE ENGINE IMPORTS
# =============================================================
# (Uit jouw yellowmind/ map)
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

# SQL gegevens
SQL_HOST = os.getenv("SQL_HOST")
SQL_USER = os.getenv("SQL_USER")
SQL_PASS = os.getenv("SQL_PASS")
SQL_DB   = os.getenv("SQL_DB")


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
# 4. SQL LOGGING
# =============================================================
def log_to_sql(question: str, answer: str):
    """
    Slaat vraag + antwoord op in yellowmind_logs.
    Mag nooit de AI breken → try/except is streng.
    """
    if not (SQL_HOST and SQL_USER and SQL_PASS and SQL_DB):
        print("🔴 SQL LOGGING SKIPPED: missing env vars")
        return

    try:
        conn = mysql.connector.connect(
            host=SQL_HOST,
            user=SQL_USER,
            password=SQL_PASS,
            database=SQL_DB,
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO yellowmind_logs (question, answer)
            VALUES (%s, %s)
            """,
            (question, answer),
        )
        conn.commit()
        cursor.close()
        conn.close()

        print("🔵 SQL LOG OK")
    except Exception as e:
        print("🔴 SQL LOGGING ERROR:", e)


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
    # 1. IDENTITY ORIGIN (vaste antwoorden)
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
        # 2. KNOWLEDGE ENGINE (eigen JSON / kennisbestand)
        # -----------------------------------------------------
        try:
            kb_answer = match_question(question, KB, language)
        except Exception as e:
            print("⚠️ knowledge engine error:", e)
            kb_answer = None

        if kb_answer:
            final_answer = kb_answer
            source = "knowledge"

        else:
            # -------------------------------------------------
            # 3. OPENAI FALLBACK (live antwoord)
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
    # 4. LOGGING NAAR SQL (ALTIJD uitvoeren)
    # ---------------------------------------------------------
    try:
        log_to_sql(question, final_answer)
    except Exception as e:
        print("⚠️ Unexpected logging error:", e)

    # ---------------------------------------------------------
    # 5. TERUG NAAR FRONTEND
    # ---------------------------------------------------------
    return {
        "answer": final_answer,
        "source": source,
    }
