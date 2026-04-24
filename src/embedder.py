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

    def song_to_text(self, song: Dict) -> str:
        energy_word = (
            "high-energy" if song["energy"] > 0.7
            else "medium-energy" if song["energy"] > 0.4
            else "low-energy"
        )
        acoustic_word = "acoustic" if song["acousticness"] > 0.6 else "electronic"
        return (
            f"A {song['mood']} {song['genre']} song. "
            f"It is {energy_word} and {acoustic_word}, "
            f"with a tempo of {song['tempo_bpm']:.0f} BPM."
        )

    def user_to_text(self, user: Dict) -> str:
        energy_word = (
            "high-energy" if user["energy"] > 0.7
            else "medium-energy" if user["energy"] > 0.4
            else "low-energy"
        )
        acoustic_word = "acoustic" if user["acousticness"] > 0.6 else "electronic"
        return (
            f"I want a {user['mood']} {user['genre']} song. "
            f"I prefer {energy_word} and {acoustic_word} music."
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
        Compute cosine similarity between the user embedding and each song
        embedding. Because both are L2-normalized, this is a dot product.
        Maps the result from [-1, 1] to [0, 1].
        """
        raw = song_embs @ user_emb  # shape (N,)
        return ((raw + 1) / 2).tolist()
