import numpy as np


class SpringEmbeddingModel:
    def __init__(self, A=None, alpha=1, beta=1, latent_dim=2, node_embeddings=None):
        """
        Initialize and empty SpringEmbeddingModel.

        Args:
            A: Adjacency matrix of the graph.
            latent_dim: Dimension of the latent space.
            node_embeddings: Node embeddings for the generative model.
        """

        self.A = A
        if A is not None:
            self.n = A.shape[0]
            self.x = np.zeros((self.n, latent_dim))
        elif node_embeddings is not None:
            self.x = node_embeddings
            self.n = node_embeddings.shape[0]
        else:
            raise ValueError(
                "Either adjacency matrix A (for model inference) or node embeddings(for generative model) must be provided."
            )
        # initialize parameters
        self.alpha = alpha
        self.beta = beta

        # intialize unweighted hamiltonian
        self.H = self.compute_H()

    def compute_H(self):
        """
        Calculate the H matrix.
        """
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                H[i, j] = self._decay_factor(i, j) * self._spring_force(i, j)

        return H

    def generate(self, expected_num_edges, allow_self_loops=False):
        """
        Generate a graph based on the current node embeddings.

        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            Adjacency matrix of the generated graph.
        """
        generated_graph = np.zeros((self.n, self.n))
        c = self._compute_density_parameter(expected_num_edges)

        for i in range(self.n):
            for j in range(self.n):
                if not allow_self_loops and i == j:
                    continue
                generated_graph[i, j] = self._generate_edges(i, j, c)

        return generated_graph

    ##### UTILITY FUNCTIONS
    def _cosine_similarity(self, a, b):
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
        cos_sim = self._cosine_similarity(self.x[i], self.x[j])
        c_ij = (1 - cos_sim) / 2
        # return 1 / (self.alpha * c_ij + 1)
        return 1 / (1 + self.alpha * c_ij)

    def _spring_force(self, i, j):
        r_ij = np.linalg.norm(self.x[i]) - np.linalg.norm(self.x[j])
        return 0.5 * (r_ij - 1) ** 2

    def _compute_density_parameter(self, expected_num_edges, allow_self_loops=False):
        """
        Compute the density parameter c based on the expected number of edges in the generated graph.

        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            c: Density parameter.
        """
        if not allow_self_loops:
            H = self.H.copy()
            np.fill_diagonal(H, 0)
        else:
            H = self.H
        c = expected_num_edges / np.sum(np.exp(-self.beta * H))
        return c

    def _generate_edges(self, i, j, c):
        """
        Generate edges between nodes i and j based on the density paramter.

        Args:
            i,j: Node indices.
            c: Density parameter.
        Returns:
            number of i -> j edges generated.
        """
        return np.random.poisson(c * np.exp(-self.beta * self.H[i, j]))


class SpringAttentionModel:
    def __init__(self, A=None, alpha=1, beta=1, latent_dim=2, X=None, Y=None):
        """
        Initialize and empty SpringEmbeddingModel.

        Args:
            A: Adjacency matrix of the graph.
            alpha: Variable spring constant weight.
            beta: Inverse temperature parameter.
            latent_dim: Dimension of the latent space.
            X: Node preference embeddings for the generative model.
            Y: Node status embeddings for the generative model.
        """

        self.A = A
        if A is not None:
            self.n = A.shape[0]
            self.x = np.zeros((self.n, latent_dim))
        elif X is not None and Y is not None:
            self.x = X
            self.y = Y
            self.n = X.shape[0]
        else:
            raise ValueError(
                "Either adjacency matrix A (for model inference) or node embeddings X, Y (for generative model) must be provided."
            )
        # initialize parameters
        self.alpha = alpha
        self.beta = beta

        # intialize unweighted hamiltonian
        self.H = self.compute_H()

    def compute_H(self):
        """
        Calculate the H matrix.
        """
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                H[i, j] = self._spring_constant(i, j) * self._spring_energy(i, j)

        return H

    def generate(self, expected_num_edges, allow_self_loops=False):
        """
        Generate a graph based on the current node embeddings.

        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            Adjacency matrix of the generated graph.
        """
        generated_graph = np.zeros((self.n, self.n))
        c = self._compute_density_parameter(expected_num_edges)

        for i in range(self.n):
            for j in range(self.n):
                if not allow_self_loops and i == j:
                    continue
                generated_graph[i, j] = self._generate_edges(i, j, c)

        return generated_graph

    ##### UTILITY FUNCTIONS
    def _cosine_similarity(self, a, b):
        """
        Calculate the cosine similarity between two vectors.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        else:
            return np.dot(a, b) / (norm_a * norm_b)

    def _spring_constant(self, i, j):
        cos_sim = self._cosine_similarity(self.x[i], self.x[j])
        # scale cosine similarity to [0, 1]?
        # c_ij = (1 + cos_sim) / 2
        # return self.alpha * c_ij
        c_ij = (1 - cos_sim) / 2
        return 1 / (1 + self.alpha * c_ij)

    def _spring_energy(self, i, j):
        s_ij = np.linalg.norm(self.y[i]) - np.linalg.norm(self.y[j])
        return 0.5 * (s_ij - 1) ** 2

    def _compute_density_parameter(self, expected_num_edges, allow_self_loops=False):
        """
        Compute the density parameter c based on the expected number of edges in the generated graph.

        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            c: Density parameter.
        """
        if not allow_self_loops:
            H = self.H.copy()
            np.fill_diagonal(H, 0)
        else:
            H = self.H
        c = expected_num_edges / np.sum(np.exp(-self.beta * H))
        return c

    def _generate_edges(self, i, j, c):
        """
        Generate edges between nodes i and j based on the density paramter.

        Args:
            i,j: Node indices.
            c: Density parameter.
        Returns:
            number of i -> j edges generated.
        """
        return np.random.poisson(c * np.exp(-self.beta * self.H[i, j]))
