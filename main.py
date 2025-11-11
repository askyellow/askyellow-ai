from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import os

# 🔹 laad de API-sleutel uit .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "AskYellow backend werkt 🎯"}


# 🔹 CORS openzetten (voor Strato / AskYellow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AskYellow backend werkt 🎯"}

@app.post("/api/vraag")
async def vraag_ai(request: Request):
    data = await request.json()
    vraag = data.get("vraag", "")
    datum = data.get("datum", "")

    if not vraag:
        return {"antwoord": "Geen vraag ontvangen."}

    prompt = f"Vandaag is het {datum}. Beantwoord accuraat en bondig: {vraag}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        antwoord = response.choices[0].message.content.strip()
        return {"antwoord": antwoord}
    except Exception as e:
        return {"antwoord": f"Fout: {e}"}
