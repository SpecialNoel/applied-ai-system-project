"""
Tests for the LLM-based preference parser (preference_parser.py).

Strategy: mock all HTTP calls so tests run without Ollama installed.
Tests are grouped into:
  1. Pure helper functions (no I/O)
  2. parse_natural_language_preference — happy paths and edge cases
  3. extract_with_reasoning — output shape and reasoning format
  4. check_ollama_running — network availability detection

Run with:
  python3 -m pytest tests/test_preference_parser.py -v
"""

# python3 -m pytest tests/test_preference_parser.py -v

import json
import pytest
from unittest.mock import patch, MagicMock

from src.preference_parser import (
    _keyword_fallback,
    _is_neutral,
    _coerce_str,
    _extract_first_object,
    _build_reasoning,
    parse_natural_language_preference,
    extract_with_reasoning,
    check_ollama_running,
    VALID_GENRES,
    VALID_MOODS,
    _DEFAULT_PREFS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ollama_response(payload: dict) -> MagicMock:
    """Return a mock requests.post response that looks like an Ollama JSON reply."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": json.dumps(payload)}
    return mock_resp


def _patch_ollama(payload: dict):
    """Context manager: patch requests.post + check_ollama_running so LLM 'works'."""
    return patch.multiple(
        "src.preference_parser",
        check_ollama_running=MagicMock(return_value=True),
        requests=_build_requests_mock(payload),
    )


def _build_requests_mock(payload: dict) -> MagicMock:
    req_mock = MagicMock()
    req_mock.post.return_value = _mock_ollama_response(payload)
    req_mock.get.return_value = MagicMock(status_code=200)
    req_mock.RequestException = Exception
    return req_mock


# ---------------------------------------------------------------------------
# 1. Pure helper functions
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    def test_workout_keywords_map_to_energetic_electronic(self):
        result = _keyword_fallback("I need music for my workout session")
        assert result["genre"] == "electronic"
        assert result["mood"] == "energetic"
        assert result["energy"] >= 0.8

    def test_study_keywords_map_to_focused_lofi(self):
        result = _keyword_fallback("background music for studying")
        assert result["genre"] == "lofi"
        assert result["mood"] == "focused"

    def test_sleep_keywords_map_to_chill_acoustic(self):
        result = _keyword_fallback("something calm to help me sleep")
        assert result["genre"] == "acoustic"
        assert result["mood"] == "chill"
        assert result["energy"] <= 0.3

    def test_sad_keywords_map_to_sad_acoustic(self):
        result = _keyword_fallback("songs for a breakup")
        assert result["genre"] == "acoustic"
        assert result["mood"] == "sad"
        assert result["valence"] <= 0.2

    def test_party_keywords_map_to_happy_pop(self):
        result = _keyword_fallback("party music to celebrate")
        assert result["genre"] == "pop"
        assert result["mood"] == "happy"

    def test_unknown_query_returns_default(self):
        result = _keyword_fallback("xyzzy")
        assert result == dict(_DEFAULT_PREFS)

    def test_returns_independent_copy(self):
        r1 = _keyword_fallback("workout")
        r2 = _keyword_fallback("workout")
        r1["genre"] = "MUTATED"
        assert r2["genre"] != "MUTATED"


class TestIsNeutral:
    def test_neutral_when_all_midpoints(self):
        assert _is_neutral({"energy": 0.5, "acousticness": 0.5, "valence": 0.5})

    def test_not_neutral_if_energy_differs(self):
        assert not _is_neutral({"energy": 0.8, "acousticness": 0.5, "valence": 0.5})

    def test_not_neutral_if_acousticness_differs(self):
        assert not _is_neutral({"energy": 0.5, "acousticness": 0.9, "valence": 0.5})

    def test_not_neutral_if_valence_differs(self):
        assert not _is_neutral({"energy": 0.5, "acousticness": 0.5, "valence": 0.1})


class TestCoerceStr:
    def test_string_passthrough(self):
        assert _coerce_str("pop") == "pop"

    def test_single_element_list_unwrapped(self):
        assert _coerce_str(["rock"]) == "rock"

    def test_empty_list_returns_empty_string(self):
        assert _coerce_str([]) == ""

    def test_non_string_converted(self):
        assert _coerce_str(42) == "42"


class TestExtractFirstObject:
    def test_plain_json_object(self):
        text = '{"genre": "pop", "mood": "happy"}'
        result = _extract_first_object(text)
        assert result["genre"] == "pop"

    def test_json_with_preamble(self):
        text = 'Here is your JSON: {"genre": "rock", "mood": "intense"}'
        result = _extract_first_object(text)
        assert result["genre"] == "rock"

    def test_json_array_returns_first_element(self):
        text = '[{"genre": "lofi", "mood": "chill"}, {"genre": "pop"}]'
        result = _extract_first_object(text)
        assert result["genre"] == "lofi"

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_first_object("not json at all")


class TestBuildReasoning:
    def test_high_energy_label(self):
        prefs = {"genre": "electronic", "mood": "energetic", "energy": 0.9, "acousticness": 0.1, "valence": 0.8}
        text = _build_reasoning("workout", prefs)
        assert "high-energy" in text
        assert "electronic" in text

    def test_low_energy_label(self):
        prefs = {"genre": "acoustic", "mood": "chill", "energy": 0.2, "acousticness": 0.9, "valence": 0.5}
        text = _build_reasoning("sleep", prefs)
        assert "low-energy" in text

    def test_mid_energy_label(self):
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.5, "valence": 0.5}
        text = _build_reasoning("something neutral", prefs)
        assert "mid-energy" in text

    def test_acoustic_descriptor(self):
        prefs = {"genre": "acoustic", "mood": "chill", "energy": 0.3, "acousticness": 0.9, "valence": 0.5}
        text = _build_reasoning("gentle music", prefs)
        assert "acoustic" in text

    def test_electronic_descriptor(self):
        prefs = {"genre": "electronic", "mood": "intense", "energy": 0.8, "acousticness": 0.1, "valence": 0.5}
        text = _build_reasoning("rave", prefs)
        assert "electronic" in text

    def test_upbeat_valence(self):
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "valence": 0.9}
        text = _build_reasoning("party", prefs)
        assert "upbeat" in text

    def test_melancholic_valence(self):
        prefs = {"genre": "acoustic", "mood": "sad", "energy": 0.2, "acousticness": 0.8, "valence": 0.1}
        text = _build_reasoning("breakup", prefs)
        assert "melancholic" in text

    def test_query_is_quoted_in_reasoning(self):
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.7, "acousticness": 0.3, "valence": 0.7}
        text = _build_reasoning("morning coffee", prefs)
        assert "morning coffee" in text


# ---------------------------------------------------------------------------
# 2. parse_natural_language_preference — with mocked HTTP
# ---------------------------------------------------------------------------

OLLAMA_UNAVAILABLE = patch(
    "src.preference_parser.check_ollama_running", return_value=False
)


class TestParseNaturalLanguagePreference:

    def _call(self, query: str, llm_payload: dict) -> dict:
        """Call the parser with a mocked Ollama that returns llm_payload."""
        mock_resp = _mock_ollama_response(llm_payload)
        with patch("src.preference_parser.check_ollama_running", return_value=True), \
             patch("src.preference_parser.requests") as mock_req:
            mock_req.post.return_value = mock_resp
            mock_req.RequestException = Exception
            return parse_natural_language_preference(query)

    # --- happy path ---

    def test_valid_llm_response_is_returned(self):
        payload = {"genre": "rock", "mood": "intense", "energy": 0.9, "acousticness": 0.1, "valence": 0.4}
        result = self._call("heavy metal vibes", payload)
        assert result["genre"] == "rock"
        assert result["mood"] == "intense"

    def test_output_always_has_required_keys(self):
        payload = {"genre": "jazz", "mood": "chill", "energy": 0.3, "acousticness": 0.7, "valence": 0.6}
        result = self._call("coffee shop music", payload)
        for key in ["genre", "mood", "energy", "acousticness", "valence"]:
            assert key in result

    def test_numeric_values_are_floats(self):
        payload = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "valence": 0.7}
        result = self._call("upbeat pop", payload)
        assert isinstance(result["energy"], float)
        assert isinstance(result["acousticness"], float)
        assert isinstance(result["valence"], float)

    # --- genre / mood coercion ---

    def test_invalid_genre_falls_back_to_default(self):
        payload = {"genre": "death-metal", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "valence": 0.7}
        result = self._call("some query", payload)
        assert result["genre"] == _DEFAULT_PREFS["genre"]

    def test_invalid_mood_falls_back_to_default(self):
        payload = {"genre": "pop", "mood": "nostalgic", "energy": 0.8, "acousticness": 0.2, "valence": 0.7}
        result = self._call("some query", payload)
        assert result["mood"] == _DEFAULT_PREFS["mood"]

    def test_genre_is_lowercased(self):
        payload = {"genre": "POP", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "valence": 0.7}
        result = self._call("upbeat", payload)
        assert result["genre"] == "pop"

    def test_mood_is_lowercased(self):
        payload = {"genre": "pop", "mood": "HAPPY", "energy": 0.8, "acousticness": 0.2, "valence": 0.7}
        result = self._call("upbeat", payload)
        assert result["mood"] == "happy"

    def test_genre_as_single_element_list_is_unwrapped(self):
        payload = {"genre": ["jazz"], "mood": "chill", "energy": 0.3, "acousticness": 0.7, "valence": 0.5}
        result = self._call("jazz cafe", payload)
        assert result["genre"] == "jazz"

    # --- numeric clamping ---

    def test_energy_above_1_is_clamped(self):
        payload = {"genre": "pop", "mood": "happy", "energy": 1.5, "acousticness": 0.2, "valence": 0.7}
        result = self._call("energetic", payload)
        assert result["energy"] == 1.0

    def test_energy_below_0_is_clamped(self):
        payload = {"genre": "pop", "mood": "happy", "energy": -0.3, "acousticness": 0.2, "valence": 0.7}
        result = self._call("still", payload)
        assert result["energy"] == 0.0

    def test_non_numeric_energy_defaults_to_0_5(self):
        payload = {"genre": "pop", "mood": "happy", "energy": "high", "acousticness": 0.2, "valence": 0.7}
        result = self._call("upbeat", payload)
        assert result["energy"] == 0.5

    # --- neutral output triggers keyword fallback ---

    def test_neutral_llm_output_triggers_keyword_fallback(self):
        """If LLM returns all 0.5 numeric values, keyword fallback should kick in."""
        payload = {"genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.5, "valence": 0.5}
        result = self._call("songs for my intense workout session", payload)
        # keyword fallback for "workout" → electronic / energetic
        assert result["genre"] == "electronic"
        assert result["mood"] == "energetic"

    # --- Ollama unavailable ---

    def test_raises_runtime_error_when_ollama_not_running(self):
        with OLLAMA_UNAVAILABLE:
            with pytest.raises(RuntimeError, match="Ollama is not running"):
                parse_natural_language_preference("test query")

    # --- network failure during request ---

    def test_returns_defaults_on_request_exception(self):
        with patch("src.preference_parser.check_ollama_running", return_value=True), \
             patch("src.preference_parser.requests") as mock_req:
            mock_req.post.side_effect = Exception("connection refused")
            mock_req.RequestException = Exception
            result = parse_natural_language_preference("any query")
        assert result == dict(_DEFAULT_PREFS)

    # --- malformed JSON from LLM ---

    def test_returns_defaults_on_json_decode_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "not valid json !!!"}
        with patch("src.preference_parser.check_ollama_running", return_value=True), \
             patch("src.preference_parser.requests") as mock_req:
            mock_req.post.return_value = mock_resp
            mock_req.RequestException = Exception
            result = parse_natural_language_preference("any query")
        assert result == dict(_DEFAULT_PREFS)

    # --- all genres and moods are accepted ---

    @pytest.mark.parametrize("genre", sorted(VALID_GENRES))
    def test_all_valid_genres_accepted(self, genre):
        payload = {"genre": genre, "mood": "happy", "energy": 0.5, "acousticness": 0.5, "valence": 0.6}
        result = self._call("test", payload)
        assert result["genre"] == genre

    @pytest.mark.parametrize("mood", sorted(VALID_MOODS))
    def test_all_valid_moods_accepted(self, mood):
        payload = {"genre": "pop", "mood": mood, "energy": 0.5, "acousticness": 0.5, "valence": 0.6}
        result = self._call("test", payload)
        assert result["mood"] == mood


# ---------------------------------------------------------------------------
# 3. extract_with_reasoning
# ---------------------------------------------------------------------------

class TestExtractWithReasoning:

    def _call(self, query: str, llm_payload: dict) -> dict:
        mock_resp = _mock_ollama_response(llm_payload)
        with patch("src.preference_parser.check_ollama_running", return_value=True), \
             patch("src.preference_parser.requests") as mock_req:
            mock_req.post.return_value = mock_resp
            mock_req.RequestException = Exception
            return extract_with_reasoning(query)

    def test_returns_preference_and_reasoning_keys(self):
        payload = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "valence": 0.9}
        result = self._call("party time", payload)
        assert "preference" in result
        assert "reasoning" in result

    def test_preference_has_all_required_fields(self):
        payload = {"genre": "lofi", "mood": "focused", "energy": 0.3, "acousticness": 0.5, "valence": 0.4}
        result = self._call("study session", payload)
        for key in ["genre", "mood", "energy", "acousticness", "valence"]:
            assert key in result["preference"]

    def test_reasoning_is_non_empty_string(self):
        payload = {"genre": "jazz", "mood": "chill", "energy": 0.3, "acousticness": 0.7, "valence": 0.6}
        result = self._call("coffee shop vibes", payload)
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0

    def test_reasoning_references_genre_and_mood(self):
        payload = {"genre": "electronic", "mood": "energetic", "energy": 0.9, "acousticness": 0.1, "valence": 0.8}
        result = self._call("intense workout", payload)
        assert "electronic" in result["reasoning"]
        assert "energetic" in result["reasoning"]


# ---------------------------------------------------------------------------
# 4. check_ollama_running
# ---------------------------------------------------------------------------

class TestCheckOllamaRunning:

    def test_returns_true_when_server_responds_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("src.preference_parser.requests.get", return_value=mock_resp):
            assert check_ollama_running() is True

    def test_returns_false_when_server_returns_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("src.preference_parser.requests.get", return_value=mock_resp):
            assert check_ollama_running() is False

    def test_returns_false_on_connection_error(self):
        import requests as real_requests
        with patch(
            "src.preference_parser.requests.get",
            side_effect=real_requests.exceptions.ConnectionError("refused"),
        ):
            assert check_ollama_running() is False

    def test_returns_false_on_timeout(self):
        import requests as real_requests
        with patch(
            "src.preference_parser.requests.get",
            side_effect=real_requests.exceptions.Timeout("timeout"),
        ):
            assert check_ollama_running() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
