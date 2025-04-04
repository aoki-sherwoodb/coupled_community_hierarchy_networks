import numpy as np


class SpringEmbeddingModel:
    def __init__(self, A, latent_dim=2):
        # network adjacency matrix (assume directed)
        self.adjacency_matrix = A
        self.n = A.shape[0]

        # initialize parameters
        self.alpha = 1
        self.beta = 1
        self.x = np.zeros((self.n, latent_dim))

    def H(self):
        """
        Calculate the H matrix.
        """
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    H[i, j] = self._decay_factor(i, j) * self._spring_force(i, j)

        return H

    ##### UTILITY FUNCTIONS
    def _cosine_similarity(a, b):
        """
        Calculate the cosine similarity between two vectors.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        else:
            return np.dot(a, b) / (norm_a * norm_b)

    def _decay_factor(self, i, j):
        c_ij = self._cosine_similarity(self.x[i], self.x[j])
        return 1 / (self.alpha * c_ij + 1)

    def _spring_force(self, i, j):
        r_ij = np.linalg.norm(self.x[i]) - np.linalg.norm(self.x[j])
        return 0.5 * (r_ij - 1) ** 2
