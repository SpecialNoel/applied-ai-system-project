#!/usr/bin/env python3
"""
Demo script for natural language preference extraction with Ollama.

Usage:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama: ollama serve
    3. Download a model: ollama pull mistral
    4. Run this script: python3 demo_natural_language.py
"""

from src.preference_parser import extract_with_reasoning, check_ollama_running
from src.recommender import load_songs, recommend_songs
from src.embedder import SongEmbedder

def main():
    # Check if Ollama is running
    if not check_ollama_running():
        print("❌ Ollama is not running!")
        print("\nTo fix:")
        print("  1. Install Ollama from: https://ollama.ai")
        print("  2. Start it with: ollama serve")
        print("  3. In another terminal, download a model: ollama pull mistral")
        print("  4. Then run this script again")
        return

    # Load songs once
    songs = load_songs("data/songs.csv")
    print(f"✓ Loaded {len(songs)} songs\n")

    # Example queries to try
    queries = [
        "songs that make me feel like I'm in a 90s indie film",
        "background music for a late night study session",
        "motivation for a tough workout",
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        try:
            # Extract preference from natural language
            result = extract_with_reasoning(query)
            prefs = result["preference"]
            reasoning = result["reasoning"]

            print(f"\nReasoning: {reasoning}")
            print(f"Extracted profile: {prefs}\n")

            # Get recommendations
            print("Top 3 recommendations (RAG mode):")
            embedder = SongEmbedder()
            recs = recommend_songs(prefs, songs, k=3, embedder=embedder)

            for i, (song, score, reasons) in enumerate(recs, 1):
                print(f"\n  #{i} {song['title']} by {song['artist']}")
                print(f"      Genre: {song['genre']} | Mood: {song['mood']}")
                print(f"      Score: {score:.2f}")

        except Exception as e:
            print(f"❌ Error: {e}")
            break

        print()


if __name__ == "__main__":
    main()

