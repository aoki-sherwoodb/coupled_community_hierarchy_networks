import numpy as np


class SpringEmbeddingModel:
    def __init__(self, A, latent_dim=2, node_embeddings=None):
        """
        Initialize and empty SpringEmbeddingModel.
        :param A: Adjacency matrix of the graph.
        :param latent_dim: Dimension of the latent space.
        :param node_embeddings: Node embeddings for a generative model. Optional.
        """
        self.adjacency_matrix = A
        self.n = A.shape[0]

        # initialize parameters
        self.alpha = 1
        self.beta = 1
        if node_embeddings is not None:
            self.x = node_embeddings
        else:
            self.x = np.zeros((self.n, latent_dim))

        # intialize unweighted hamiltonian
        self.H = self.compute_H()

    def compute_H(self):
        """
        Calculate the H matrix.
        """
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    H[i, j] = self._decay_factor(i, j) * self._spring_force(i, j)

        return H

    def generate(self, expected_num_edges):
        """
        Generate a graph based on the current node embeddings.
        :param expected_num_edges: Expected number of edges in the generated graph.
        :return: Adjacency matrix of the generated graph.
        """
        generated_graph = np.zeros((self.n, self.n))
        c = self._compute_density_parameter(expected_num_edges)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                generated_graph[i, j] = self._generate_edges(i, j, c)

        return generated_graph

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

    def _compute_density_parameter(self, expected_num_edges):
        """
        Compute the density parameter c based on the expected number of edges in the generated graph.
        :param expected_num_edges: Expected number of edges.
        :return: Density parameter c.
        """
        c = expected_num_edges / np.sum(np.exp(-self.beta * self.H))
        return c

    def _generate_edges(self, i, j, c):
        """
        Generate edges between nodes i and j based on the decay factor.
        :param i: Node i.
        :param j: Node j.
        :param c: Density parameter.
        :return: number of i -> j edges generated.
        """
        return np.random.poisson(c * -self.beta * self.H[i, j])
