from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    genre: str
    mood: str
    energy: float
    acousticness: float
    valence: float

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py

    Pass a SongEmbedder instance to enable RAG mode, where genre and mood
    are matched semantically instead of with exact string comparison.
    """
    def __init__(self, songs: List[Dict], embedder=None):
        """Initialize the recommender with a list of song dicts."""
        self.songs = songs
        self.embedder = embedder
        self._song_embeddings = None
        if embedder is not None:
            self._song_embeddings = embedder.embed_songs(songs)

    def _get_semantic_scores(self, user: Dict) -> List[float]:
        """
        Batch-compute a semantic similarity score in [0, 1] for every song
        against the user profile. Called once per recommend() call.
        """
        user_emb = self.embedder.embed_user(user)
        return self.embedder.cosine_similarities(user_emb, self._song_embeddings)

    def score_song(self, user: Dict, song: Dict, semantic_score: float = None) -> Tuple[float, List[str]]:
        '''
        Return (score, reasons) for a single song.

        When semantic_score is provided (RAG mode), the combined 0.55-point
        genre + mood component is replaced by 0.55 * semantic_score so that
        near-synonym moods and genres still earn partial credit.
        When semantic_score is None (classic mode), exact string matching is
        used as before.
        '''
        score = 0.0
        reasons = []

        if semantic_score is not None:
            # RAG: continuous semantic similarity replaces binary genre + mood match
            semantic_points = 0.55 * semantic_score
            score += semantic_points
            reasons.append(
                f"Semantic similarity (mood + genre) {semantic_score:.2f} (+{semantic_points:.2f})"
            )
        else:
            # Classic: exact string matching
            mood_points = 0.35 if user['mood'] == song['mood'] else 0
            score += mood_points
            if mood_points:
                reasons.append(f"Mood matches your favorite '{song['mood']}' (+{mood_points:.2f})")
            else:
                reasons.append(f"Mood '{song['mood']}' does not match your favorite '{user['mood']}' (+0.00)")
            mood_points = 0

            genre_points = 0.20 if user['genre'] == song['genre'] else 0
            score += genre_points
            if genre_points:
                reasons.append(f"Genre matches your favorite '{song['genre']}' (+{genre_points:.2f})")
            else:
                reasons.append(f"Genre '{song['genre']}' does not match your favorite '{user['genre']}' (+0.00)")

        # Numeric features — identical in both modes
        energy_points = 0.20 * (1 - abs(song['energy'] - user['energy']))
        score += energy_points
        reasons.append(f"Energy {song['energy']:.2f} vs your target {user['energy']:.2f} (+{energy_points:.2f})")

        valence_points = 0.10 * (1 - abs(song['valence'] - user['valence']))
        score += valence_points
        reasons.append(f"Valence {song['valence']:.2f} vs your target {user['valence']:.2f} (+{valence_points:.2f})")

        tempo_points = 0.08 * (song['tempo_bpm'] - 40) / 160
        score += tempo_points
        reasons.append(f"Tempo {song['tempo_bpm']:.0f} BPM (+{tempo_points:.2f})")

        acousticness_points = 0.05 * (1 - abs(song['acousticness'] - user['acousticness']))
        score += acousticness_points
        reasons.append(f"Acousticness {song['acousticness']:.2f} vs your target {user['acousticness']:.2f} (+{acousticness_points:.2f})")

        danceability_points = 0.02 * song['danceability']
        score += danceability_points
        reasons.append(f"Danceability {song['danceability']:.2f} (+{danceability_points:.2f})")

        return score, reasons

    def recommend(self, user: Dict, k: int = 5) -> List[Dict]:
        '''
        Return the top-k recommended song dicts for the user.
        Uses RAG scoring if an embedder was provided at init, else classic.
        '''
        if self.embedder is not None:
            sem_scores = self._get_semantic_scores(user)
            scored = [
                (song, self.score_song(user, song, semantic_score=sem)[0])
                for song, sem in zip(self.songs, sem_scores)
            ]
        else:
            scored = [(song, self.score_song(user, song)[0]) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                'id': int(row['id']),
                'title': row['title'],
                'artist': row['artist'],
                'genre': row['genre'],
                'mood': row['mood'],
                'energy': float(row['energy']),
                'tempo_bpm': float(row['tempo_bpm']),
                'valence': float(row['valence']),
                'danceability': float(row['danceability']),
                'acousticness': float(row['acousticness']),
            })
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, embedder=None) -> List[Tuple[Dict, float, List[str]]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Pass a SongEmbedder as embedder to enable RAG mode.
    """
    recommender = Recommender(songs, embedder=embedder)
    if embedder is not None:
        sem_scores = recommender._get_semantic_scores(user_prefs)
        scored = [
            (song, *recommender.score_song(user_prefs, song, semantic_score=sem))
            for song, sem in zip(songs, sem_scores)
        ]
    else:
        scored = [(song, *recommender.score_song(user_prefs, song)) for song in songs]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
