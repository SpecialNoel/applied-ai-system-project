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
from typing import Dict, Optional

# Valid values for structured preferences (should match your song catalog)
VALID_GENRES = {"pop", "rock", "lofi", "jazz", "electronic", "acoustic"}
VALID_MOODS = {"happy", "chill", "energetic", "intense", "sad", "focused"}

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"


def check_ollama_running() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def _extract_first_object(text: str) -> Dict:
    """Extract the first JSON object from text that may be an array or have extra wrapping."""
    # Find the outermost JSON structure (array or object)
    array_start = text.find("[")
    object_start = text.find("{")

    if array_start != -1 and (object_start == -1 or array_start < object_start):
        # Response is a JSON array — take the first element
        end = text.rfind("]") + 1
        parsed = json.loads(text[array_start:end])
        return parsed[0] if isinstance(parsed, list) else parsed
    else:
        # Response is a plain JSON object
        end = text.rfind("}") + 1
        return json.loads(text[object_start:end])


def parse_natural_language_preference(
    user_query: str,
    song_examples: Optional[list] = None,
    model: str = "tinyllama",
) -> Dict:
    """
    Convert natural language query into structured music preferences using Ollama.

    Args:
        user_query: Freeform text like "songs that make me feel like I'm in a 90s indie film"
        song_examples: Optional list of songs to help the model understand the catalog
        model: Ollama model to use (default: mistral)

    Returns:
        Dictionary with keys: genre, mood, energy, acousticness, valence
        All numeric values normalized to [0, 1]
    """

    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Please start it with: ollama serve\n"
            "And download a model with: ollama pull mistral"
        )

    # Build context about available genres and moods
    context = f"""You are a music preference extraction assistant. Your job is to convert natural language
queries into structured music preferences.

Available genres: {', '.join(sorted(VALID_GENRES))}
Available moods: {', '.join(sorted(VALID_MOODS))}

Return ONLY valid JSON with these fields (all numeric values MUST be floats between 0.0 and 1.0):
- genre: one of the available genres (best match) - STRING
- mood: one of the available moods (best match) - STRING
- energy: 0.0=calm, 1.0=high-energy - FLOAT ONLY
- acousticness: 0.0=electronic, 1.0=acoustic - FLOAT ONLY
- valence: 0.0=sad/serious, 1.0=happy/positive - FLOAT ONLY

IMPORTANT: energy, acousticness, and valence MUST be numbers like 0.5, 0.7, etc. NOT text."""

    prompt = f"""{context}

User query: "{user_query}"

Return ONLY valid JSON, no explanation:"""

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        response_text = response.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Ollama request failed: {e}")
        return {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.5,
            "acousticness": 0.3,
            "valence": 0.5,
        }

    # Parse JSON from response
    try:
        prefs = _extract_first_object(response_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse response as JSON: {response_text}")
        return {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.5,
            "acousticness": 0.3,
            "valence": 0.5,
        }

    # Validate and normalize values
    prefs["genre"] = prefs.get("genre", "pop").lower()
    if prefs["genre"] not in VALID_GENRES:
        prefs["genre"] = "pop"  # fallback

    prefs["mood"] = prefs.get("mood", "happy").lower()
    if prefs["mood"] not in VALID_MOODS:
        prefs["mood"] = "happy"  # fallback

    # Clamp numeric values to [0, 1]
    for key in ["energy", "acousticness", "valence"]:
        try:
            val = float(prefs.get(key, 0.5))
        except (ValueError, TypeError):
            # If it's not a valid float, use default
            val = 0.5
        prefs[key] = max(0.0, min(1.0, val))

    return prefs

def extract_with_reasoning(user_query: str, model: str = "tinyllama") -> Dict:
    """
    Parse preference and return both the structured prefs and reasoning from Ollama.

    Returns:
        {
            "preference": {...},
            "reasoning": "why we extracted these values"
        }
    """

    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Please start it with: ollama serve\n"
            "And download a model with: ollama pull mistral"
        )

    context = f"""You are a music preference extraction assistant. Your job is to convert natural language
queries into structured music preferences.

Available genres: {', '.join(sorted(VALID_GENRES))}
Available moods: {', '.join(sorted(VALID_MOODS))}

Return ONLY valid JSON with:
- preference: {{"genre": STRING, "mood": STRING, "energy": FLOAT, "acousticness": FLOAT, "valence": FLOAT}}
- reasoning: explain your mapping in one sentence (STRING)

IMPORTANT: energy, acousticness, and valence MUST be numbers like 0.5, 0.7, etc. NOT text like "medium"."""

    prompt = f"""{context}

User query: "{user_query}"

Return ONLY valid JSON:"""

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        response_text = response.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Ollama request failed: {e}")
        prefs = {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.5,
            "acousticness": 0.3,
            "valence": 0.5,
        }
        return {
            "preference": prefs,
            "reasoning": "Error connecting to Ollama, returning defaults"
        }

    try:
        result = _extract_first_object(response_text)
        prefs = result.get("preference", result)  # fallback: treat whole object as prefs
    except json.JSONDecodeError:
        print(f"Warning: Could not parse response: {response_text}")
        prefs = {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.5,
            "acousticness": 0.3,
            "valence": 0.5,
        }
        return {
            "preference": prefs,
            "reasoning": "Error parsing response"
        }

    # Validate
    prefs["genre"] = prefs.get("genre", "pop").lower()
    if prefs["genre"] not in VALID_GENRES:
        prefs["genre"] = "pop"

    prefs["mood"] = prefs.get("mood", "happy").lower()
    if prefs["mood"] not in VALID_MOODS:
        prefs["mood"] = "happy"

    for key in ["energy", "acousticness", "valence"]:
        try:
            val = float(prefs.get(key, 0.5))
        except (ValueError, TypeError):
            val = 0.5
        prefs[key] = max(0.0, min(1.0, val))

    return {
        "preference": prefs,
        "reasoning": result.get("reasoning", "") if 'result' in locals() else ""
    }
