from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import uvicorn

load_dotenv()

# Load key
key = os.getenv("OPENAI_API_KEY")
print("🔑 API key:", "OK" if key else "MISSING")

app = FastAPI()

# CORS (Strato → Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ROOT test
@app.get("/")
def home():
    return {"message": "AskYellow backend actief 🎯"}


# AI route
@app.post("/api/vraag")
async def vraag_ai(data: dict):
    vraag = data.get("vraag", "")
    datum = data.get("datum", "")

    if not vraag:
        return {"antwoord": "Geen vraag ontvangen."}

    vraag_lower = vraag.lower()

    # Special hard-coded AskYellow answers
    live_triggers = [
        "ben je live",
        "sta je live",
        "ben jij live",
        "ben je al live",
        "sta je al live",
        "ben je echt online",
        "ben jij echt online"
    ]
    if any(t in vraag_lower for t in live_triggers):
        return {
            "antwoord": (
                "Ja! 🙌 AskYellow AI Beta staat nu echt live op askyellow.nl. "
                "Ik draai via onze eigen AskYellow-server en leer elke dag bij. "
                "Je kunt me gewoon gebruiken als jouw persoonlijke antwoordmachine. 💛"
            )
        }

    # AskYellow Character Prompt V2 (professioneel + veilig)
    system_prompt = """
Je bent AskYellow AI, de antwoordmachine van AskYellow.nl.
Je geeft heldere, rustige en eerlijke antwoorden.
Je gebruikt GEEN zinnen zoals:
- “mijn kennis stopt in …”
- “ik ben niet live”
- “ik ben een AI-model”
- “mijn training is tot …”
- “ik heb geen toegang tot internet”

In plaats daarvan:
- Geef uitleg alsof je een behulpzame assistent bent.
- Zeg alleen bij snelle ontwikkelingen dat informatie recent kan zijn veranderd.
- Gebruik de taal van de gebruiker (NL of EN).
- Geef GEEN gevaarlijke claims of medische/financiële zekerheden.
- Blijf vriendelijk, direct en merk-consistent.

Onze stijl:
- kalm
- eerlijk
- duidelijk
- menselijk
"""

    user_content = f"Vandaag is het {datum}. De gebruiker vraagt: {vraag}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content}
            ],
        )

        antwoord = response.choices[0].message.content.strip()

        # Confidence / sanity check
        lower = antwoord.lower()
        if any(x in lower for x in ["ik weet het niet", "geen idee", "kan niet vinden"]):
            antwoord += (
                "\n\n(📌 Tip: Dit onderwerp verandert snel. Controleer belangrijke informatie altijd even extra.)"
            )

        return {"antwoord": antwoord}

    except Exception as e:
        # Safety fallback
        return {
            "antwoord": (
                "Ik kan op dit moment geen live antwoord ophalen. "
                "Probeer het over een paar seconden opnieuw. ⚠️"
            )
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

