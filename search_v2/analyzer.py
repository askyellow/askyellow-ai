import json
import re
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You analyze Dutch user input for an e-commerce search engine.

Return ONLY valid JSON with these fields:
- intent (string)
- category (string or null)
- new_constraints (object)
- is_negative (boolean)
- missing_info (array)
- should_refine (boolean)
- refine_question (string or null)
- wants_to_buy_now (boolean)
- budget_style (string or null)

General rules:
- Output JSON only. No explanations.
- Detect product search intent.
- Extract relevant keywords and price_max if mentioned.
- If the input is like "nee", "geen", or another negative answer, set is_negative = true.
- Do not invent products.

Intent:
- product_search = clear buying intent
- assisted_search = user wants help choosing the right type/specification before buying
- general_question = general informational question

Intent rules:
- Classify based on intent pattern, not only product name.
- If the conversation is already in assisted_search mode, do not switch to product_search unless wants_to_buy_now is true.

Buying intent:
- wants_to_buy_now = true only if the user clearly wants to shop now, for example asking for products, links, prices, or using phrases like "ik zoek", "kopen", "bestellen".

Concrete product rule:
- If the user explicitly mentions a concrete product noun, treat the product as already known.
- Do NOT ask which product within the category they mean.
- Instead, ask about preferences, constraints, or features.

Categories:
When possible, assign one of these high-level categories:
- huishoudelijk
- beeld_en_geluid
- sport
- gaming
- mobiliteit
- gereedschap
- mode
- beauty_verzorging
- algemeen

Category rules:
- Choose the closest matching category.
- Do not invent new category names.

Constraints:
- Extract price_max if mentioned.
- budget_style = "low" if the user says things like "zo goedkoop mogelijk", "goedkoop", "budget", or "laagste prijs".
- Infer implicit environment constraints when obvious:
  - badkamer, woonkamer, slaapkamer -> environment: indoor
  - tuin, plantenbak, terras -> environment: outdoor

Missing info:
- missing_info should only contain information that is truly not yet provided.
- Do not mark something as missing if it is already explicitly mentioned in the user input.

Refinement:
- should_refine = true only if one extra question would significantly improve search quality.
- refine_question must be exactly one short Dutch question.
- Ask about only one high-impact attribute at a time.
- Never ask about budget again if price_max is already known.
- If the input is negative, do not invent a new refinement question.

Examples of assisted_search:
- "wat voor verf moet ik gebruiken"
- "welke boormachine heb ik nodig"
- "wat voor matras past bij mij"
"""

def ai_analyze_input(user_input: str, state: dict | None = None):
    context = ""

    if state:
        context = f"""
Huidige zoekstate:
intent: {state.get("intent")}
category: {state.get("category")}
constraints: {state.get("constraints")}
refinement_done: {state.get("refinement_done")}
"""

    user_message = context + "\nLaatste input:\n" + user_input

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    content = re.sub(r"```json", "", content)
    content = re.sub(r"```", "", content).strip()

    return json.loads(content)

def ai_generate_refinement_question(state: dict | None) -> str:
    if not isinstance(state, dict):
        return "Kun je iets meer details geven zodat ik beter kan helpen?"

    category = state.get("category")
    price_max = state.get("constraints", {}).get("price_max")

    prompt = f"""
Je bent een slimme e-commerce assistent.

De gebruiker zoekt naar:
Categorie: {category}
Maximale prijs: {price_max}

Stel EXACT 1 korte, natuurlijke vervolgvraag
die de zoekresultaten significant verfijnt.

Vraag niet opnieuw naar budget.
Geen uitleg.
Alleen de vraag.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def ai_generate_targeted_question(state: dict, missing_info: list, original_input: str) -> str:
    prompt = f"""
Je bent een slimme e-commerce assistent.

Originele vraag van de gebruiker:
"{original_input}"

Huidige categorie: {state.get("category")}
Bekende informatie: {state.get("constraints")}
Ontbrekende informatie: {missing_info}

Belangrijk:
- Stel GEEN vraag over iets dat al expliciet in de originele vraag staat.
- Herhaal geen gebruik of toepassing als die al genoemd is.
- Stel EXACT 1 korte, natuurlijke vraag over de belangrijkste ontbrekende eigenschap.
- Geen uitleg.
- Alleen de vraag.
"""


    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

