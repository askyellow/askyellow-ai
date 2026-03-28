from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from datetime import datetime
from zoneinfo import ZoneInfo

def build_time_context() -> str:
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    return (
        f"Vandaag is het {now.strftime('%d %B %Y')} (Europe/Amsterdam). "
        "Deze tijdsinformatie is actueel, zeker en leidend. "
        "Behandel deze datum als feit. "
        "Zeg nooit dat je de huidige datum, tijd of actualiteit niet weet. "
        "Relatieve termen zoals 'vandaag', 'gisteren', 'morgen', "
        "'vorig jaar', 'dit jaar', 'recent' en 'afgelopen jaarwisseling' "
        "moeten vanuit deze datum worden geïnterpreteerd."
    )
def day_part() -> str:
    hour = datetime.now(ZoneInfo("Europe/Amsterdam")).hour
    if hour < 12:
        return "goedemorgen"
    elif hour < 18:
        return "goedemiddag"
    else:
        return "goedenavond"

def greeting() -> str:
    return day_part().capitalize()

def build_llm_time_hint() -> str:
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    if now.hour < 12:
        dagdeel = "ochtend"
    elif now.hour < 18:
        dagdeel = "middag"
    else:
        dagdeel = "avond"

    return (
        f"Het is nu {dagdeel}. "
        "Gebruik dit dagdeel natuurlijk in toon of begroeting als dat relevant is."
    )

def get_logical_date():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).date()
