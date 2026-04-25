from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
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
    genre: str
    mood: str
    energy: float
    acousticness: float
    valence: float

# Scoring weights — must sum to 1.0 in each mode.
# RAG mode:     SEMANTIC(0.55) + ENERGY(0.20) + VALENCE(0.10) + TEMPO(0.08) + ACOUSTICNESS(0.05) + DANCEABILITY(0.02)
# Classic mode: MOOD(0.35) + GENRE(0.20) + ENERGY(0.20) + VALENCE(0.10) + TEMPO(0.08) + ACOUSTICNESS(0.05) + DANCEABILITY(0.02)
W_SEMANTIC     = 0.55
W_MOOD         = 0.35
W_GENRE        = 0.20
W_ENERGY       = 0.20
W_VALENCE      = 0.10
W_TEMPO        = 0.08
W_ACOUSTICNESS = 0.05
W_DANCEABILITY = 0.02


class Recommender:
    """
    Recommendation engine supporting two scoring modes:
      - Classic: exact string match on genre and mood.
      - RAG: semantic similarity via a SongEmbedder (pass at init).
    """
    def __init__(self, songs: List[Dict], embedder: Optional[object] = None):
        self.songs = songs
        self.embedder = embedder
        self._song_embeddings = None
        if embedder is not None:
            self._song_embeddings = embedder.embed_songs(songs)

    def _get_semantic_scores(self, user: Dict) -> List[float]:
        user_emb = self.embedder.embed_user(user)
        return self.embedder.cosine_similarities(user_emb, self._song_embeddings)

    def score_song(self, user: Dict, song: Dict, semantic_score: float = None) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if semantic_score is not None:
            semantic_points = W_SEMANTIC * semantic_score
            score += semantic_points
            reasons.append(
                f"Semantic similarity (mood + genre) {semantic_score:.2f} (+{semantic_points:.2f})"
            )
        else:
            mood_points = W_MOOD if user['mood'] == song['mood'] else 0
            score += mood_points
            if mood_points:
                reasons.append(f"Mood matches your favorite '{song['mood']}' (+{mood_points:.2f})")
            else:
                reasons.append(f"Mood '{song['mood']}' does not match your favorite '{user['mood']}' (+0.00)")

            genre_points = W_GENRE if user['genre'] == song['genre'] else 0
            score += genre_points
            if genre_points:
                reasons.append(f"Genre matches your favorite '{song['genre']}' (+{genre_points:.2f})")
            else:
                reasons.append(f"Genre '{song['genre']}' does not match your favorite '{user['genre']}' (+0.00)")

        energy_points = W_ENERGY * (1 - abs(song['energy'] - user['energy']))
        score += energy_points
        reasons.append(f"Energy {song['energy']:.2f} vs your target {user['energy']:.2f} (+{energy_points:.2f})")

        valence_points = W_VALENCE * (1 - abs(song['valence'] - user['valence']))
        score += valence_points
        reasons.append(f"Valence {song['valence']:.2f} vs your target {user['valence']:.2f} (+{valence_points:.2f})")

        preferred_tempo = 40 + 160 * user['energy']
        tempo_points = W_TEMPO * (1 - abs(song['tempo_bpm'] - preferred_tempo) / 160)
        score += tempo_points
        reasons.append(f"Tempo {song['tempo_bpm']:.0f} BPM vs preferred {preferred_tempo:.0f} BPM (+{tempo_points:.2f})")

        acousticness_points = W_ACOUSTICNESS * (1 - abs(song['acousticness'] - user['acousticness']))
        score += acousticness_points
        reasons.append(f"Acousticness {song['acousticness']:.2f} vs your target {user['acousticness']:.2f} (+{acousticness_points:.2f})")

        preferred_danceability = user['energy']
        danceability_points = W_DANCEABILITY * (1 - abs(song['danceability'] - preferred_danceability))
        score += danceability_points
        reasons.append(f"Danceability {song['danceability']:.2f} vs preferred {preferred_danceability:.2f} (+{danceability_points:.2f})")

        return score, reasons

    def recommend(self, user: Dict, k: int = 5) -> List[Dict]:
        """Return the top-k recommended song dicts for the user."""
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
    """Load songs from a CSV file and return a list of dicts."""
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


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    embedder=None,
) -> List[Tuple[Dict, float, List[str]]]:
    """Functional wrapper around Recommender. Supports both classic and RAG modes."""
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
