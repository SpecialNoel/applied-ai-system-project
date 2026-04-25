from typing import Dict, List
import numpy as np
from fastembed import TextEmbedding


class SongEmbedder:
    """
    Converts songs and user profiles into text descriptions,
    then embeds them using a lightweight ONNX-based model so that
    semantically similar moods and genres (e.g. 'intense' vs
    'energetic') score close together instead of binary 0/1.

    Uses fastembed (ONNX Runtime backend — no PyTorch required).
    The model is downloaded once and cached locally (~40 MB).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)

    @staticmethod
    def _energy_word(energy: float) -> str:
        if energy > 0.7:
            return "high-energy"
        if energy > 0.4:
            return "medium-energy"
        return "low-energy"

    @staticmethod
    def _acoustic_word(acousticness: float) -> str:
        return "acoustic" if acousticness > 0.6 else "electronic"

    def song_to_text(self, song: Dict) -> str:
        return (
            f"A {song['mood']} {song['genre']} song. "
            f"It is {self._energy_word(song['energy'])} and {self._acoustic_word(song['acousticness'])}, "
            f"with a tempo of {song['tempo_bpm']:.0f} BPM."
        )

    def user_to_text(self, user: Dict) -> str:
        return (
            f"I want a {user['mood']} {user['genre']} song. "
            f"I prefer {self._energy_word(user['energy'])} and {self._acoustic_word(user['acousticness'])} music."
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings of shape (N, D)."""
        vecs = np.array(list(self.model.embed(texts)))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)

    def embed_songs(self, songs: List[Dict]) -> np.ndarray:
        """Embed all songs and return their embedding matrix."""
        return self.embed([self.song_to_text(s) for s in songs])

    def embed_user(self, user: Dict) -> np.ndarray:
        """Embed a user profile and return its embedding vector."""
        return self.embed([self.user_to_text(user)])[0]

    def cosine_similarities(self, user_emb: np.ndarray, song_embs: np.ndarray) -> List[float]:
        """
        Dot product of L2-normalized vectors equals cosine similarity.
        Remapped from [-1, 1] to [0, 1].
        """
        raw = song_embs @ user_emb  # shape (N,)
        return ((raw + 1) / 2).tolist()
