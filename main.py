from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import os

# -------------------------
# 1. LOAD API KEY
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# -------------------------
# 2. FASTAPI SETUP
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# 3. LOAD MODULES
# -------------------------
def load_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read() + "\n\n"
    except:
        return ""

def load_character():
    base = "yellowmind/"
    system_prompt = ""

    # CORE
    system_prompt += load_file(base + "core/core_identity.txt")
    system_prompt += load_file(base + "core/mission.txt")
    system_prompt += load_file(base + "core/values.txt")
    system_prompt += load_file(base + "core/introduction_rules.txt")
    system_prompt += load_file(base + "core/communication_baseline.txt")

    # PARENTS
    system_prompt += load_file(base + "parents/parent_profile_brigitte.txt")
    system_prompt += load_file(base + "parents/parent_profile_dennis.txt")
    system_prompt += load_file(base + "parents/parent_mix_logic.txt")

    # BEHAVIOUR
    system_prompt += load_file(base + "behaviour/behaviour_rules.txt")
    system_prompt += load_file(base + "behaviour/boundaries_safety.txt")
    system_prompt += load_file(base + "behaviour/escalation_rules.txt")
    system_prompt += load_file(base + "behaviour/uncertainty_handling.txt")
    system_prompt += load_file(base + "behaviour/user_types.txt")

    # KNOWLEDGE
    system_prompt += load_file(base + "knowledge/knowledge_sources.txt")
    system_prompt += load_file(base + "knowledge/askyellow_site_rules.txt")
    system_prompt += load_file(base + "knowledge/product_rules.txt")
    system_prompt += load_file(base + "knowledge/no_hallucination_rules.txt")
    system_prompt += load_file(base + "knowledge/limitations.txt")

    # SYSTEM (master prompt)
    system_prompt += load_file(base + "system/yellowmind_master_prompt_v2.txt")

    return system_prompt


# -------------------------
# 4. DETECT TONE MODULE
# -------------------------
def detect_tone(user_input):
    text = user_input.lower()
    base = "yellowmind/tone/"

    if any(x in text for x in ["huil", "moeilijk", "ik weet niet", "help", "verdriet"]):
        return load_file(base + "empathy_mode.txt")

    if any(x in text for x in ["api", "error", "code", "liquid", "script", "dns", "bug"]):
        return load_file(base + "tech_mode.txt")

    if any(x in text for x in ["kort", "snel", "opsomming"]):
        return load_file(base + "concise_mode.txt")

    if any(x in text for x in ["verhaal", "creative", "sprookje", "fantasie"]):
        return load_file(base + "storytelling_mode.txt")

    if any(x in text for x in ["askyellow", "branding", "logo", "stijl"]):
        return load_file(base + "branding_mode.txt")

    return ""


# -------------------------
# 5. REQUEST MODEL (correct Swagger + correct JSON)
# -------------------------
class AskInput(BaseModel):
    question: str


# -------------------------
# 6. MAIN AI ENDPOINT (/ask)
# -------------------------
@app.post("/ask")
async def ask(data: AskInput):

    question = data.question

    system_prompt = load_character() + detect_tone(question)

    # NEW OPENAI SDK (2024+)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    answer = response.output_text

    return {"answer": answer}


# -------------------------
# 7. STATUS CHECK
# -------------------------
@app.get("/")
async def root():
    return {"status": "Yellowmind v2.0 draait 🤖💛"}
