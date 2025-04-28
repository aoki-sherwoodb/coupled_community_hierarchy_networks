import numpy as np


def normalized_cosine_similarity(x_i, x_j):
    """Compute cosine similarity between vectors x_i and x_j normalized to [0, 1]."""

    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)

    if norm_i < 1e-8 or norm_j < 1e-8:
        return 0.5  # catch division by zero

    cos_sim = np.dot(x_i, x_j) / (norm_i * norm_j)
    # Normalize to [0, 1]
    return (cos_sim + 1) / 2


def spring_constant(x_i, x_j, alpha):
    cos_sim = normalized_cosine_similarity(x_i, x_j)  # already normalized to [0, 1]
    return alpha * cos_sim


def spring_energy(x_i, x_j):
    s_ij = np.linalg.norm(x_i) - np.linalg.norm(x_j)
    return 0.5 * (s_ij - 1) ** 2


def matrix_cosine_sim(X: np.ndarray):
    norm = np.linalg.norm(X, axis=1)
    dot = X @ X.T
    norm = np.outer(norm, norm)
    cosine_sim = dot / (norm + 1e-9)
    return np.clip(cosine_sim, -1, 1)


def scaled_cosine_sim(X: np.ndarray, k: int = 1):
    """
    Compute the k-th power of the cosine similarities of vectors in X scaled to [0, 1].
    """
    cosine_sim = matrix_cosine_sim(X)
    scaled = (cosine_sim + 1) / 2
    return np.power(scaled, k)