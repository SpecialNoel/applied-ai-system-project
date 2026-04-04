"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 
    print(f'Loaded songs: {len(songs)}')

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8,
                  'acousticness': 0.1, 'valence': 0.2}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 50)
    print("  TOP RECOMMENDATIONS")
    print("=" * 50)
    for i, (song, score, reasons) in enumerate(recommendations, start=1):
        print(f"\n#{i}  {song['title']} by {song['artist']}")
        print(f"    Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"    Score: {score:.2f}")
        print("    Why:")
        for reason in reasons:
            print(f"      - {reason}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
