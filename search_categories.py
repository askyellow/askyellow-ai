SEARCH_CATEGORIES = {
    "beeld_en_geluid": {
        "label": "Beeld & Geluid",
        "facets": ["type", "gebruik", "budget", "formaat"]
    },
    "huishouden": {
        "label": "Huishouden",
        "facets": [
            "type",        # wat voor apparaat
            "gebruik",     # waarvoor / context
            "budget",      # prijsindicatie
            "voorkeur"     # merk of speciale wensen (optioneel)
        ],
        "max_questions": 2
    },
    "mode": {
        "label": "Mode",
        "facets": ["type", "stijl", "geslacht", "budget"]
    },
    "gaming": {
        "label": "Gaming",
        "facets": ["platform", "type", "budget", "gebruik"]
    },
    "speelgoed": {
        "label": "Speelgoed",
        "facets": ["leeftijd", "type", "interesse", "budget"]
    },
    "sport": {
        "label": "Sport & Outdoor",
        "facets": ["sport", "gebruik", "niveau", "budget"]
    },
    "beauty_verzorging": {
        "label": "Beauty & Verzorging",
        "facets": ["type", "gebruik", "huid_haar", "budget"]
    },
    "mode_accessoires": {
        "label": "Mode & Accessoires",
        "facets": ["type", "stijl", "gebruik", "budget"]
    }
}

PROMPT_HUISHOUDEN = {"""De gebruiker zoekt een product in de categorie Huishouden.
De vraag is nog te algemeen om een concreet advies te geven.

Relevante keuzedimensies zijn:
- type
- gebruik
- budget
- voorkeur

Stel maximaal 2 korte, vriendelijke vervolgvragen
om de keuze te verfijnen.
Gebruik normale spreektaal.
Stel geen ja/nee-vragen.
Geen opsommingen, geen lijstjes.

Doel: de gebruiker helpen kiezen, niet pushen."""
}