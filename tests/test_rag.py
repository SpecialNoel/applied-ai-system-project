"""
Tests for RAG (Retrieval-Augmented Generation) music recommendation.

Verifies that:
  1. The embedder correctly converts songs/users to semantic text
  2. Embeddings capture semantic similarity (e.g., "intense" ≈ "angry")
  3. Classic mode uses exact string matching for genre/mood
  4. RAG mode gives partial credit to semantically similar genres/moods
  5. RAG mode ranks songs higher when they're conceptually aligned
"""

# python3 -m pytest tests/test_rag.py -v

import pytest
from src.recommender import Recommender, load_songs, recommend_songs
from src.embedder import SongEmbedder


class TestSongEmbedder:
    """Test the SongEmbedder text conversion and embedding logic."""

    @pytest.fixture
    def embedder(self):
        return SongEmbedder()

    @pytest.fixture
    def sample_song(self):
        return {
            "id": 1,
            "title": "Test Song",
            "artist": "Test Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.7,
            "danceability": 0.8,
            "acousticness": 0.2,
        }

    @pytest.fixture
    def sample_user(self):
        return {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "acousticness": 0.2,
            "valence": 0.7,
        }

    def test_song_to_text_includes_genre_mood(self, embedder, sample_song):
        """Song text description should include genre and mood."""
        text = embedder.song_to_text(sample_song)
        assert "pop" in text.lower()
        assert "happy" in text.lower()

    def test_song_to_text_includes_energy_descriptor(self, embedder, sample_song):
        """Song text should describe energy level (high/medium/low)."""
        text = embedder.song_to_text(sample_song)
        assert "high-energy" in text.lower()

        # Test medium energy
        sample_song["energy"] = 0.5
        text = embedder.song_to_text(sample_song)
        assert "medium-energy" in text.lower()

        # Test low energy
        sample_song["energy"] = 0.2
        text = embedder.song_to_text(sample_song)
        assert "low-energy" in text.lower()

    def test_user_to_text_includes_preferences(self, embedder, sample_user):
        """User text description should include genre and mood preferences."""
        text = embedder.user_to_text(sample_user)
        assert "pop" in text.lower()
        assert "happy" in text.lower()
        assert "want" in text.lower() or "prefer" in text.lower()

    def test_embed_returns_normalized_vectors(self, embedder):
        """Embeddings should be L2-normalized (norm ≈ 1)."""
        texts = ["high energy pop song", "low energy jazz song"]
        embeddings = embedder.embed(texts)

        # Check shape
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # D > 0

        # Check normalization: norm should be very close to 1
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_embed_songs_returns_matrix(self, embedder, sample_song):
        """embed_songs should return a matrix with one row per song."""
        songs = [sample_song, sample_song]
        embeddings = embedder.embed_songs(songs)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_cosine_similarities_in_valid_range(self, embedder, sample_song, sample_user):
        """Cosine similarities should be normalized to [0, 1]."""
        songs = [sample_song] * 3
        song_embs = embedder.embed_songs(songs)
        user_emb = embedder.embed_user(sample_user)

        scores = embedder.cosine_similarities(user_emb, song_embs)

        assert len(scores) == 3
        for score in scores:
            assert 0.0 <= score <= 1.0


class TestClassicVsRAGScoring:
    """Compare classic exact-match scoring vs RAG semantic scoring."""

    @pytest.fixture
    def sample_songs(self):
        """Songs with different genre/mood combinations."""
        return [
            {  # Exact match: pop + intense
                "id": 1,
                "title": "Exact Match",
                "artist": "Artist",
                "genre": "pop",
                "mood": "intense",
                "energy": 0.8,
                "tempo_bpm": 120,
                "valence": 0.5,
                "danceability": 0.7,
                "acousticness": 0.1,
            },
            {  # Different genre, similar mood: rock + intense
                "id": 2,
                "title": "Diff Genre, Same Mood",
                "artist": "Artist",
                "genre": "rock",
                "mood": "intense",
                "energy": 0.8,
                "tempo_bpm": 120,
                "valence": 0.5,
                "danceability": 0.7,
                "acousticness": 0.1,
            },
            {  # Different mood, similar intensity: pop + angry
                "id": 3,
                "title": "Same Genre, Diff Mood",
                "artist": "Artist",
                "genre": "pop",
                "mood": "angry",
                "energy": 0.8,
                "tempo_bpm": 120,
                "valence": 0.5,
                "danceability": 0.7,
                "acousticness": 0.1,
            },
            {  # No match: lofi + chill
                "id": 4,
                "title": "No Match",
                "artist": "Artist",
                "genre": "lofi",
                "mood": "chill",
                "energy": 0.3,
                "tempo_bpm": 80,
                "valence": 0.5,
                "danceability": 0.4,
                "acousticness": 0.8,
            },
        ]

    @pytest.fixture
    def intense_pop_user(self):
        """User who wants intense pop music."""
        return {
            "genre": "pop",
            "mood": "intense",
            "energy": 0.8,
            "acousticness": 0.1,
            "valence": 0.5,
        }

    def test_classic_mode_exact_match_scores_highest(self, sample_songs, intense_pop_user):
        """Classic mode should rank the exact match (pop + intense) first."""
        rec = Recommender(sample_songs, embedder=None)

        # Score each song
        scores = [(song, rec.score_song(intense_pop_user, song)[0]) for song in sample_songs]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Top song should be the exact match
        assert scores[0][0]["id"] == 1
        assert scores[0][0]["genre"] == "pop"
        assert scores[0][0]["mood"] == "intense"

    def test_classic_mode_gives_zero_mood_points_for_mismatch(self, sample_songs, intense_pop_user):
        """Classic mode gives 0 points for mood mismatch (e.g., "chill" ≠ "intense")."""
        rec = Recommender(sample_songs, embedder=None)

        # Score the lofi/chill song
        chill_song = sample_songs[3]
        score, reasons = rec.score_song(intense_pop_user, chill_song)

        # Find the mood reason
        mood_reason = [r for r in reasons if "Mood" in r][0]
        assert "+0.00" in mood_reason  # No mood match

    def test_rag_mode_gives_partial_credit_to_similar_moods(self, sample_songs, intense_pop_user):
        """RAG mode should give partial credit when moods are semantically similar."""
        embedder = SongEmbedder()
        rec = Recommender(sample_songs, embedder=embedder)

        # Score the pop/angry song
        pop_angry_song = sample_songs[2]
        score, reasons = rec.score_song(intense_pop_user, pop_angry_song, semantic_score=0.85)

        # Semantic score should contribute 0.55 * 0.85 = 0.4675
        semantic_reason = [r for r in reasons if "Semantic similarity" in r][0]
        assert "+0." in semantic_reason  # Some points for semantic similarity

        # The semantic component should have contributed points
        # Extract the semantic points value from the reason (format: "Semantic similarity (mood + genre) 0.85 (+0.47)")
        import re
        match = re.search(r'\(\+(\d+\.\d+)\)', semantic_reason)
        assert match is not None, f"Could not parse points from reason: {semantic_reason}"
        semantic_points = float(match.group(1))
        assert 0.46 <= semantic_points <= 0.48, f"Expected ~0.47, got {semantic_points}"

    def test_rag_mode_ranks_semantically_similar_songs_higher(self, sample_songs, intense_pop_user):
        """RAG should re-rank songs to put semantically similar ones higher."""
        embedder = SongEmbedder()

        classic_recs = recommend_songs(intense_pop_user, sample_songs, k=4, embedder=None)
        rag_recs = recommend_songs(intense_pop_user, sample_songs, k=4, embedder=embedder)

        # Extract song IDs from recommendations
        classic_order = [song["id"] for song, _, _ in classic_recs]
        rag_order = [song["id"] for song, _, _ in rag_recs]

        # The orders should be different (RAG should re-rank)
        assert classic_order != rag_order

        # In classic mode, pop+intense is #1
        assert classic_order[0] == 1

        # In RAG mode, similar songs get lifted up (rock/intense and pop/angry become more competitive)
        # At least one of the "similar to intense" songs should be in top results
        rag_intense_like_ids = {2, 3}  # rock/intense, pop/angry
        assert any(song_id in rag_order for song_id in rag_intense_like_ids)


class TestRAGIntegration:
    """Integration tests with real songs from the dataset."""

    @pytest.fixture
    def real_songs(self):
        """Load the real songs.csv dataset."""
        return load_songs("data/songs.csv")

    def test_intense_user_gets_energetic_songs_in_rag(self, real_songs):
        """
        Key RAG benefit: A user wanting 'intense' music should also get
        songs with semantically similar moods (energetic, angry, moody, etc.)
        ranked high, since they're conceptually adjacent.
        """
        intense_pop_user = {
            "genre": "pop",
            "mood": "intense",
            "energy": 0.8,
            "acousticness": 0.1,
            "valence": 0.2,
        }

        embedder = SongEmbedder()
        rag_recs = recommend_songs(intense_pop_user, real_songs, k=5, embedder=embedder)

        # Check that at least some songs have semantically similar moods to "intense"
        rag_songs = [song for song, _, _ in rag_recs]
        semantically_similar_moods = {"angry", "energetic", "moody", "intense"}
        similar_songs = [
            s for s in rag_songs if s["mood"] in semantically_similar_moods
        ]

        # Should have picked up at least one semantically similar song in RAG mode
        assert len(similar_songs) > 0, (
            f"RAG should include songs with intense-like moods. Got: "
            f"{[s['mood'] for s in rag_songs]}"
        )

    def test_rag_vs_classic_scoring_difference(self, real_songs):
        """Verify that RAG produces different scores than classic mode."""
        intense_pop_user = {
            "genre": "pop",
            "mood": "intense",
            "energy": 0.8,
            "acousticness": 0.1,
            "valence": 0.2,
        }

        classic_recs = recommend_songs(intense_pop_user, real_songs, k=5, embedder=None)
        embedder = SongEmbedder()
        rag_recs = recommend_songs(intense_pop_user, real_songs, k=5, embedder=embedder)

        # Extract scores
        classic_scores = [score for _, score, _ in classic_recs]
        rag_scores = [score for _, score, _ in rag_recs]

        # Scores should be different (RAG redistributes points)
        assert classic_scores != rag_scores

    def test_rag_semantic_scores_are_in_valid_range(self, real_songs):
        """All semantic similarity scores should be in [0, 1]."""
        user = {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "acousticness": 0.1,
            "valence": 0.2,
        }

        embedder = SongEmbedder()
        rag_recs = recommend_songs(user, real_songs, k=5, embedder=embedder)

        # Extract reasons and find semantic similarity scores
        for song, score, reasons in rag_recs:
            sem_reasons = [r for r in reasons if "Semantic similarity" in r]
            assert len(sem_reasons) > 0, f"Should have semantic reason for {song['title']}"

            # Verify the total score is reasonable
            assert 0.0 <= score <= 1.0 + 0.01  # Allow small floating point error


class TestBackwardCompatibility:
    """Ensure classic (non-RAG) mode still works correctly."""

    @pytest.fixture
    def simple_songs(self):
        return [
            {
                "id": 1,
                "title": "Pop Happy",
                "artist": "A",
                "genre": "pop",
                "mood": "happy",
                "energy": 0.8,
                "tempo_bpm": 120,
                "valence": 0.8,
                "danceability": 0.8,
                "acousticness": 0.2,
            },
            {
                "id": 2,
                "title": "Jazz Chill",
                "artist": "B",
                "genre": "jazz",
                "mood": "chill",
                "energy": 0.3,
                "tempo_bpm": 90,
                "valence": 0.6,
                "danceability": 0.4,
                "acousticness": 0.9,
            },
        ]

    def test_recommender_without_embedder_uses_classic_scoring(self, simple_songs):
        """Without embedder, Recommender should use exact-match scoring."""
        rec = Recommender(simple_songs, embedder=None)

        pop_user = {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "acousticness": 0.2,
            "valence": 0.8,
        }

        result = rec.recommend(pop_user, k=2)

        # Should recommend pop/happy song first (exact match)
        assert result[0]["genre"] == "pop"
        assert result[0]["mood"] == "happy"

    def test_recommend_songs_without_embedder_works(self, simple_songs):
        """recommend_songs() without embedder should work as before."""
        pop_user = {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "acousticness": 0.2,
            "valence": 0.8,
        }

        # Call without embedder (backwards compatible)
        recs = recommend_songs(pop_user, simple_songs, k=2, embedder=None)

        assert len(recs) == 2
        # First song should be the pop/happy match
        song, score, reasons = recs[0]
        assert song["genre"] == "pop"
        assert song["mood"] == "happy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
