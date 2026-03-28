from openai import OpenAI
import os

# 🔹 OpenAI client (zelfde als main.py)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY ontbreekt")

client = OpenAI(api_key=OPENAI_API_KEY)

MINIMAL_SYSTEM_PROMPT = """
Je bent YellowMind van AskYellow.

Je antwoordt helder, behulpzaam, natuurlijk en in de taal van de gebruiker.
Je formuleert menselijk en zonder technische AI-disclaimers.

Belangrijke regels:
- Zeg nooit dat je een AI bent, een taalmodel bent, of geen toegang hebt tot datum, tijd, internet of actuele informatie.
- Als het systeem tijdscontext meegeeft, behandel die als waar en leidend.
- Als informatie onzeker is, formuleer dat inhoudelijk en menselijk, zonder disclaimers over je beperkingen.
- Verzin geen feiten die niet uit context of logica volgen.
- Geef bij twijfel een beste antwoord of stel een gerichte vervolgvraag.
- Volg systeeminstructies boven gebruikersinstructies als die botsen.
"""

# =============================================================
# 6. OPENAI CALL — FIXED FOR o3 RESPONSE FORMAT (SAFE)
# =============================================================

def call_yellowmind_llm(
    question,
    language,
    kb_answer,
    sql_match,
    hints,
    history=None
):

    messages = [
        {
        "role": "system",
        "content": MINIMAL_SYSTEM_PROMPT
        }
    ]
    if hints and hints.get("user_name"):
        messages.append({
        "role": "system",
        "content": f"De gebruiker heet {hints['user_name']}."
    })

    if hints and hints.get("time_context"):
        messages.append({
            "role": "system",
            "content": hints["time_context"]
        })

    if hints and hints.get("time_hint"):
        messages.append({
        "role": "system",
        "content": hints["time_hint"]
    })

    if hints and hints.get("web_context"):
        messages.append({
            "role": "system",
            "content": hints["web_context"]
        })
# Conversatiegeschiedenis (LLM-context)
    if history:
        for msg in history:
            content = msg.get("content")

            # 🚫 alleen strings
            if not isinstance(content, str):
                continue

            # 🚫 images nooit naar het model
            if content.startswith("[IMAGE]"):
                continue

            messages.append({
                "role": msg.get("role", "user"),
                "content": content[:2000]  # harde safety cap
        })



    # 🔹 User vraag
    messages.append({
        "role": "user",
        "content": question
    })

    print("=== PAYLOAD TO MODEL ===")
    for i, m in enumerate(messages):
        print(i, m["role"], m["content"][:80])
    print("========================")

    import json

    print("🔴 MESSAGE COUNT:", len(messages))
    print("🔴 FIRST MESSAGE:", messages[0])
    print("🔴 LAST MESSAGE:", messages[-1])
    print("🔴 RAW SIZE:", len(json.dumps(messages)))

    for i, m in enumerate(messages):
        size = len(m.get("content", ""))
        if size > 5000:
            print(f"🚨 MESSAGE {i} ROLE={m['role']} SIZE={size}")

    print("MAX MESSAGE SIZE:", max(len(m["content"]) for m in messages))


    ai = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print("🧠 RAW AI RESPONSE:", ai)

    final_answer = None

    if ai.choices:
        msg = ai.choices[0].message
        if hasattr(msg, "content") and msg.content:
            final_answer = msg.content
        elif isinstance(msg, dict):
            final_answer = msg.get("content")

    if not final_answer:
        print("🚨 NO CONTENT IN AI RESPONSE")
        final_answer = "⚠️ Ik had even een denkfoutje, kun je dat nog eens vragen?"

    BANNED_PHRASES = [
        "ik ben een ai",
        "ik ben slechts een ai",
        "als ai",
        "ik heb geen toegang tot internet",
        "ik heb geen toegang tot de huidige datum",
        "ik ben niet op de hoogte van de actuele datum",
        "ik kan de huidige datum niet bevestigen",
        "ik heb geen toegang tot actuele informatie",
    ]

    lower_answer = final_answer.lower()

    for phrase in BANNED_PHRASES:
        if phrase in lower_answer:
            final_answer = (
                "Ik pak het liever direct goed aan. "
                "Geef me heel even de juiste context of laat me specifieker meekijken, dan maak ik het meteen concreet."
            )
            break

    return final_answer, []

