from src.recommender import Recommender

_SONGS = [
    {
        "id": 1, "title": "Test Pop Track", "artist": "Test Artist",
        "genre": "pop", "mood": "happy",
        "energy": 0.8, "tempo_bpm": 120, "valence": 0.9,
        "danceability": 0.8, "acousticness": 0.2,
    },
    {
        "id": 2, "title": "Chill Lofi Loop", "artist": "Test Artist",
        "genre": "lofi", "mood": "chill",
        "energy": 0.4, "tempo_bpm": 80, "valence": 0.6,
        "danceability": 0.5, "acousticness": 0.9,
    },
]

_POP_USER = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2, "valence": 0.9}
_LOFI_USER = {"genre": "lofi", "mood": "chill", "energy": 0.4, "acousticness": 0.9, "valence": 0.6}


def test_recommend_returns_correct_count():
    rec = Recommender(_SONGS)
    assert len(rec.recommend(_POP_USER, k=2)) == 2


def test_recommend_ranks_matching_song_first():
    rec = Recommender(_SONGS)
    results = rec.recommend(_POP_USER, k=2)
    assert results[0]["genre"] == "pop"
    assert results[0]["mood"] == "happy"


def test_recommend_ranks_lofi_first_for_lofi_user():
    rec = Recommender(_SONGS)
    results = rec.recommend(_LOFI_USER, k=2)
    assert results[0]["genre"] == "lofi"


def test_score_song_returns_float_and_reasons():
    rec = Recommender(_SONGS)
    score, reasons = rec.score_song(_POP_USER, _SONGS[0])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert len(reasons) > 0


def test_score_song_matching_scores_higher_than_mismatching():
    rec = Recommender(_SONGS)
    score_match, _ = rec.score_song(_POP_USER, _SONGS[0])   # pop/happy user, pop/happy song
    score_miss, _  = rec.score_song(_POP_USER, _SONGS[1])   # pop/happy user, lofi/chill song
    assert score_match > score_miss
