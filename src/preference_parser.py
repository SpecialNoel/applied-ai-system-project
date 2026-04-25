"""
Natural language preference parser using Ollama (local LLM).

Converts freeform user queries into structured music preferences.
Examples:
  - "songs that make me feel like I'm in a 90s indie film"
  - "motivation for a tough workout"
  - "background music for late night studying"

Requires: ollama installed and a model running (e.g., `ollama pull mistral`)
"""

import json
import requests
from typing import Dict

VALID_GENRES = {"pop", "rock", "lofi", "jazz", "electronic", "acoustic"}
VALID_MOODS  = {"happy", "chill", "energetic", "intense", "sad", "focused"}

OLLAMA_API = "http://localhost:11434/api/generate"

_DEFAULT_PREFS: Dict = {
    "genre": "pop",
    "mood": "happy",
    "energy": 0.5,
    "acousticness": 0.3,
    "valence": 0.5,
}

_KEYWORD_RULES = [
    ({"workout", "gym", "run", "running", "exercise", "pump", "lifting", "cardio", "tough"},
     {"genre": "electronic", "mood": "energetic", "energy": 0.9, "acousticness": 0.1, "valence": 0.7}),
    ({"study", "studying", "focus", "concentrate", "work", "coding", "reading"},
     {"genre": "lofi", "mood": "focused", "energy": 0.3, "acousticness": 0.5, "valence": 0.4}),
    ({"sleep", "relax", "relaxing", "calm", "peaceful", "meditation"},
     {"genre": "acoustic", "mood": "chill", "energy": 0.2, "acousticness": 0.9, "valence": 0.5}),
    ({"sad", "heartbreak", "cry", "crying", "lonely", "breakup", "melancholy"},
     {"genre": "acoustic", "mood": "sad", "energy": 0.2, "acousticness": 0.7, "valence": 0.1}),
    ({"party", "dance", "club", "hype", "celebrate"},
     {"genre": "pop", "mood": "happy", "energy": 0.9, "acousticness": 0.1, "valence": 0.9}),
    ({"indie", "90s", "nostalgic", "film", "cinematic", "vintage"},
     {"genre": "acoustic", "mood": "chill", "energy": 0.5, "acousticness": 0.6, "valence": 0.6}),
    ({"jazz", "coffee", "cafe", "morning"},
     {"genre": "jazz", "mood": "chill", "energy": 0.3, "acousticness": 0.7, "valence": 0.6}),
    ({"night", "late", "midnight", "dark", "moody"},
     {"genre": "lofi", "mood": "chill", "energy": 0.3, "acousticness": 0.5, "valence": 0.3}),
]


def check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _keyword_fallback(query: str) -> Dict:
    words = set(query.lower().split())
    for keywords, prefs in _KEYWORD_RULES:
        if words & keywords:
            return dict(prefs)
    return dict(_DEFAULT_PREFS)


def _is_neutral(prefs: Dict) -> bool:
    return prefs["energy"] == 0.5 and prefs["acousticness"] == 0.5 and prefs["valence"] == 0.5


def _coerce_str(value) -> str:
    """Coerce LLM output to string, unwrapping single-element lists."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _extract_first_object(text: str) -> Dict:
    """Extract the first JSON object from text that may be an array or have extra wrapping."""
    array_start = text.find("[")
    object_start = text.find("{")

    if array_start != -1 and (object_start == -1 or array_start < object_start):
        end = text.rfind("]") + 1
        parsed = json.loads(text[array_start:end])
        return parsed[0] if isinstance(parsed, list) else parsed
    else:
        end = text.rfind("}") + 1
        return json.loads(text[object_start:end])


def parse_natural_language_preference(user_query: str, model: str = "mistral") -> Dict:
    """
    Convert a natural language query into structured music preferences using Ollama.

    Returns a dict with keys: genre, mood, energy, acousticness, valence.
    Falls back to keyword matching if the LLM returns neutral values, or to
    defaults if Ollama is unavailable.
    """
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Then download a model with: ollama pull mistral"
        )

    context = (
        f"Genres: {', '.join(sorted(VALID_GENRES))}. "
        f"Moods: {', '.join(sorted(VALID_MOODS))}. "
        "Return JSON with exactly these keys: genre, mood, energy, acousticness, valence. "
        "energy: 0.0=very calm, 1.0=very energetic. "
        "acousticness: 0.0=electronic, 1.0=acoustic. "
        "valence: 0.0=sad, 1.0=happy."
    )
    prompt = f"{context}\nQuery: \"{user_query}\"\nJSON:"

    try:
        response = requests.post(
            OLLAMA_API,
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=120,
        )
        response.raise_for_status()
        response_text = response.json()["response"].strip()
    except requests.RequestException as e:
        print(f"Warning: Ollama request failed: {e}")
        return dict(_DEFAULT_PREFS)

    try:
        raw = _extract_first_object(response_text)
    except json.JSONDecodeError:
        print("Could not understand that query, using default preferences.")
        return dict(_DEFAULT_PREFS)

    raw["genre"] = _coerce_str(raw.get("genre", _DEFAULT_PREFS["genre"])).lower()
    if raw["genre"] not in VALID_GENRES:
        raw["genre"] = _DEFAULT_PREFS["genre"]

    raw["mood"] = _coerce_str(raw.get("mood", _DEFAULT_PREFS["mood"])).lower()
    if raw["mood"] not in VALID_MOODS:
        raw["mood"] = _DEFAULT_PREFS["mood"]

    for key in ["energy", "acousticness", "valence"]:
        try:
            val = float(raw.get(key, 0.5))
        except (ValueError, TypeError):
            val = 0.5
        raw[key] = max(0.0, min(1.0, val))

    result = {k: raw[k] for k in ["genre", "mood", "energy", "acousticness", "valence"]}
    if _is_neutral(result):
        result = _keyword_fallback(user_query)
    return result


def _build_reasoning(query: str, prefs: Dict) -> str:
    energy_desc   = "high-energy" if prefs["energy"] > 0.6 else ("low-energy" if prefs["energy"] < 0.4 else "mid-energy")
    acoustic_desc = "acoustic" if prefs["acousticness"] > 0.6 else "electronic"
    valence_desc  = "upbeat" if prefs["valence"] > 0.6 else ("melancholic" if prefs["valence"] < 0.4 else "neutral")
    return (
        f'"{query}" interpreted as {prefs["mood"]} {prefs["genre"]} — '
        f"{energy_desc}, {acoustic_desc}, {valence_desc} tone."
    )


def extract_with_reasoning(user_query: str, model: str = "mistral") -> Dict:
    """
    Parse a natural language query and return both the structured preference
    and a human-readable reasoning summary.

    Returns: {"preference": {...}, "reasoning": "..."}
    """
    prefs = parse_natural_language_preference(user_query, model=model)
    return {
        "preference": prefs,
        "reasoning": _build_reasoning(user_query, prefs),
    }
