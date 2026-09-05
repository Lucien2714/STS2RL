"""Locate bundled STS2 JSON data and normalize its collection names."""

from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "json"


DATA_TYPE_ALIASES = {
    "act": "acts",
    "ascension": "ascensions",
    "card": "cards",
    "character": "characters",
    "enchantment": "enchantments",
    "encounter": "encounters",
    "event": "events",
    "intent": "intents",
    "keyword": "keywords",
    "modifier": "modifiers",
    "monster": "monsters",
    "orb": "orbs",
    "potion": "potions",
    "power": "powers",
    "relic": "relics",
}


def normalize_data_type(data_type: str) -> str:
    """Normalize singular or alias data names to their JSON collection name."""
    normalized = data_type.strip().lower()
    return DATA_TYPE_ALIASES.get(normalized, normalized)
