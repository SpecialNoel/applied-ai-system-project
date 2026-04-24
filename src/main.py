"""
Command line runner for the Music Recommender Simulation.

Runs the same user profile through two modes side-by-side:
  Classic — binary exact-match on genre and mood
  RAG     — semantic similarity via sentence-transformers embeddings

Supports natural language preference input using Ollama (local LLM, free).
"""

import sys
from .recommender import load_songs, recommend_songs
from .embedder import SongEmbedder
from .preference_parser import extract_with_reasoning, check_ollama_running

# Check if Ollama is available
HAS_OLLAMA = check_ollama_running()


def print_recommendations(title: str, recommendations: list) -> None:
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)
    for i, (song, score, reasons) in enumerate(recommendations, start=1):
        print(f"\n#{i}  {song['title']} by {song['artist']}")
        print(f"    Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"    Score: {score:.2f}")
        print("    Why:")
        for reason in reasons:
            print(f"      - {reason}")
    print()


def get_user_preference():
    """
    Interactively get user preference from natural language or use default.
    """
    print("\n" + "=" * 55)
    print("  MUSIC PREFERENCE INPUT")
    print("=" * 55)

    options = "\nOptions:"
    if HAS_OLLAMA:
        options += "\n  1. Enter a natural language query (e.g., 'songs for a 90s indie vibe')"
        options += "\n  2. Use default profile (pop, intense, 0.8 energy)"
        default_choice = "2"
    else:
        options += "\n  1. Use default profile (pop, intense, 0.8 energy)"
        options += "\n\n  💡 Tip: Install Ollama to enable natural language queries!"
        options += "\n     Download from: https://ollama.ai"
        default_choice = "1"

    print(options)

    choice = input(f"\nChoose {('1 or 2' if HAS_OLLAMA else '1')} (default: {default_choice}): ").strip()

    if HAS_OLLAMA and choice == "1":
        query = input("\nDescribe the music you want: ").strip()
        if not query:
            print("Empty query, using default profile.")
            return {
                "genre": "pop",
                "mood": "intense",
                "energy": 0.8,
                "acousticness": 0.1,
                "valence": 0.2,
            }

        print("\nParsing your preference with Ollama...")
        try:
            result = extract_with_reasoning(query)
            prefs = result["preference"]
            reasoning = result["reasoning"]

            print(f"\n✓ Extracted preference:")
            print(f"  Query: '{query}'")
            print(f"  Reasoning: {reasoning}")
            print(f"  Profile: {prefs}")

            return prefs
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Using default profile instead.")
            return {
                "genre": "pop",
                "mood": "intense",
                "energy": 0.8,
                "acousticness": 0.1,
                "valence": 0.2,
            }
    else:
        return {
            "genre": "pop",
            "mood": "intense",
            "energy": 0.8,
            "acousticness": 0.1,
            "valence": 0.2,
        }


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs.")

    # Get user preference (natural language or default)
    user_prefs = get_user_preference()
    print(f"\nUser profile: {user_prefs}")

    # --- Classic mode (no embedder) ---
    classic_recs = recommend_songs(user_prefs, songs, k=5)
    print_recommendations("CLASSIC MODE  (exact genre + mood match)", classic_recs)

    # --- RAG mode ---
    print("Loading sentence-transformer model (downloads once on first run)...")
    embedder = SongEmbedder()
    print("Model ready.\n")

    rag_recs = recommend_songs(user_prefs, songs, k=5, embedder=embedder)
    print_recommendations("RAG MODE  (semantic genre + mood similarity)", rag_recs)


if __name__ == "__main__":
    main()
