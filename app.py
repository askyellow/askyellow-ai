from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

load_dotenv()

# Check API key
key = os.getenv("OPENAI_API_KEY")
print("✅ API key loaded successfully!" if key else "❌ API key NOT found!")

# Init FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI setup
openai.api_key = key

@app.post("/api/vraag")
async def vraag_ai(data: dict):
    vraag = data.get("vraag", "")
    if not vraag:
        return {"antwoord": "Geen vraag ontvangen."}
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": vraag}],
        )
        antwoord = response.choices[0].message.content.strip()
        return {"antwoord": antwoord}
    except Exception as e:
        return {"antwoord": f"Fout: {e}"}

# Run only locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
