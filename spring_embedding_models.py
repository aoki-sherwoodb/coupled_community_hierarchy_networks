import numpy as np
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from NetworkEmbeddingModel import NetworkEmbeddingModel
import utils


class SpringEmbeddingModel(NetworkEmbeddingModel):
    def __init__(
        self,
        adj_matrix,
        embedding_dim=2,
        alpha=1.0,
        beta=1.0,
        allow_self_loops=False,
    ):
        """
        Initialize a SpringEmbeddingModel for inference.

        Parameters:
        -----------
        adj_matrix : numpy.ndarray
            The adjacency matrix of the network
        embedding_dim : int
            Dimension of the embedding vectors
        alpha : float
            Weight parameter for the spring force term
        """
        self.adj_matrix = np.array(adj_matrix)
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.allow_self_loops = allow_self_loops

        self.num_nodes = self.adj_matrix.shape[0]
        self.edge_list = []

        # Convert adjacency matrix to edge list for efficient computation
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):  # Only upper triangular (undirected)
                if self.adj_matrix[i, j] > 0:
                    self.edge_list.append((i, j))

        # Initialize embeddings randomly
        self.embeddings = np.random.normal(0, 1, size=(self.num_nodes, embedding_dim))

        # Prior parameters (from Hoff et al. 2002)
        self.prior_mean = np.zeros(embedding_dim)
        self.prior_var = 1.0  # Prior variance for each dimension

        # MCMC parameters
        self.proposal_var = 0.1  # Will be adapted during burn-in
        self.accepted = 0
        self.total_proposals = 0

    def log_prior(self, node_idx):
        """Compute multivariate normal log prior probability for a node's embedding."""
        log_p = multivariate_normal.logpdf(
            self.embeddings[node_idx],
            mean=self.prior_mean,
            cov=self.prior_var * np.eye(self.embedding_dim),
        )
        return log_p

    def energy_edge(self, i, j):
        """Compute energy for edge between nodes i and j."""
        x_i = self.embeddings[i]
        x_j = self.embeddings[j]

        # Compute Hamiltonian
        return utils.spring_constant(x_i, x_j, self.alpha) * utils.spring_energy(
            x_i, x_j
        )

    def hamiltonian(self):
        """Compute total energy of the system (Hamiltonian)."""
        total = 0.0
        for i, j in self.edge_list:
            total += self.energy_edge(i, j)
        return total

    def log_likelihood_edge(self, i, j):
        """
        Compute log likelihood for edge between nodes i and j
        using Boltzmann distribution with inverse temperature beta.
        """
        # Calculate energy
        energy = self.energy_edge(i, j)

        # Convert energy to log probability using Boltzmann distribution
        # P(A_ij=1|X) ∝ exp(-beta * energy)
        log_p = -self.beta * energy

        return log_p

    def log_likelihood_node(self, i):
        """Compute log likelihood for all outgoing edges from node i."""
        log_p = 0.0

        for j in range(self.num_nodes):
            if not self.allow_self_loops and j == i:
                continue

            if self.adj_matrix[i, j] > 0:
                log_p += self.log_likelihood_edge(i, j)

        return log_p

    def log_posterior_node(self, node_idx):
        """Compute log posterior for a node's embedding."""
        return self.log_prior(node_idx) + self.log_likelihood_node(node_idx)

    def propose_new_position(self, node_idx):
        """Propose a new position for node_idx using random walk MH."""
        # Random walk proposal
        proposal = self.embeddings[node_idx] + np.random.normal(
            0, np.sqrt(self.proposal_var), size=self.embedding_dim
        )

        return proposal

    def metropolis_hastings_step(self, node_idx):
        """Perform one Metropolis-Hastings step for a node."""
        # Current log posterior
        current_log_posterior = self.log_posterior_node(node_idx)

        # Save current position
        current_pos = self.embeddings[node_idx].copy()

        # Propose new position
        proposed_pos = self.propose_new_position(node_idx)

        # Temporarily set the node to proposed position
        self.embeddings[node_idx] = proposed_pos

        # Compute new log posterior
        proposed_log_posterior = self.log_posterior_node(node_idx)

        # Compute acceptance probability
        log_accept_ratio = proposed_log_posterior - current_log_posterior

        # Accept or reject
        self.total_proposals += 1
        if np.log(np.random.uniform(0, 1)) < log_accept_ratio:
            # Accept the proposal
            self.accepted += 1
            return True
        else:
            # Reject the proposal and revert to the previous position
            self.embeddings[node_idx] = current_pos
            return False

    def adapt_proposal_variance(self, acceptance_rate):
        """Adapt the proposal variance to achieve target acceptance rate."""
        target_rate = 0.234  # Optimal for multidimensional targets

        # Increase variance if acceptance rate is too high
        if acceptance_rate > target_rate:
            self.proposal_var *= 1.1
        # Decrease variance if acceptance rate is too low
        else:
            self.proposal_var *= 0.9

    def run_mcmc(
        self,
        n_iterations=10000,
        burn_in=2000,
        adapt_interval=100,
        thin=50,
        annealing=True,
        beta_start=0.1,
        beta_end=10,
    ):
        """
        Run MCMC inference as described in Section 3 of the paper.

        Parameters:
        -----------
        n_iterations : int
            Total number of MCMC iterations
        burn_in : int
            Number of burn-in iterations to discard
        adapt_interval : int
            Interval for adapting proposal variance
        thin : int
            Thinning factor for collecting samples
        annealing : bool
            Whether to use simulated annealing (gradually increase beta)
        beta_start : float
            Starting inverse temperature (only used if annealing=True)
        beta_end : float
            Ending inverse temperature (only used if annealing=True)
            If None, will use self.beta as the ending temperature

        Returns:
        --------
        samples : list
            List of embedding samples after burn-in and thinning
        energies : list
            List of energy values for each sample
        """
        samples = []
        energies = []

        # Setup annealing schedule if requested
        if annealing:
            if beta_end is None:
                beta_end = self.beta

            # Only anneal during burn-in
            if burn_in > 0:
                beta_schedule = np.linspace(beta_start, beta_end, burn_in)
            else:
                beta_schedule = [beta_end] * n_iterations
        else:
            beta_schedule = [self.beta] * n_iterations

        for iteration in tqdm(range(n_iterations)):
            # Update beta for annealing if in burn-in phase
            if annealing and iteration < len(beta_schedule):
                self.beta = beta_schedule[iteration]

            # Perform MH steps for each node
            for node_idx in range(self.num_nodes):
                self.metropolis_hastings_step(node_idx)

            # Calculate current total energy
            current_energy = self.hamiltonian()

            # Adapt proposal variance during burn-in
            if (
                iteration < burn_in
                and iteration > 0
                and iteration % adapt_interval == 0
            ):
                acceptance_rate = self.accepted / self.total_proposals
                self.adapt_proposal_variance(acceptance_rate)

                tqdm.write(
                    f"Iteration {iteration}, Acceptance rate: {acceptance_rate:.4f}, "
                    f"Proposal variance: {self.proposal_var:.6f}, "
                    f"Beta: {self.beta:.4f}, Energy: {current_energy:.4f}"
                )

                # Reset counters
                self.accepted = 0
                self.total_proposals = 0

            # Collect samples after burn-in with thinning
            if iteration >= burn_in and iteration % thin == 0:
                samples.append(self.embeddings.copy())
                energies.append(current_energy)

        # Report final acceptance rate
        final_acceptance_rate = self.accepted / max(1, self.total_proposals)
        print(f"Final acceptance rate: {final_acceptance_rate:.4f}")
        print(f"Final energy: {self.hamiltonian():.4f}")

        return samples, energies

    def get_map_estimate(self, samples=None, energies=None):
        """Get maximum a posteriori (MAP) estimate from MCMC samples."""
        if samples is None or energies is None:
            return self.embeddings.copy()

        # Find sample with minimum energy (maximum probability)
        min_energy_idx = np.argmin(energies)
        return samples[min_energy_idx]

    def fit(self):
        """
        Fit the model to the data using MCMC inference.
        """
        # Run MCMC
        samples, energies = self.run_mcmc()

        # Get MAP estimate
        self.embeddings = self.get_map_estimate(samples, energies)

        return self.embeddings

    def predict(self, i, j):
        """
        Calculate the probability of a directed edge i -> j conditioned on the existence of an edge i <-> j.
        """
        h_ij = self.energy_edge(i, j)
        h_ji = self.energy_edge(j, i)
        return np.exp(-self.beta * h_ij) / np.sum(
            np.exp(-self.beta * h_ji) + np.exp(-self.beta * h_ij)
        )


class SpringEmbeddingGenerativeModel(NetworkEmbeddingModel):
    def __init__(self, embeddings, alpha=1, beta=1, allow_self_loops=False):
        """
        Initialize a generative SpringEmbeddingModel with the given parameters.

        Args:
            embeddings: Node embeddings for the generative model.
            alpha: Variable spring constant weight.
            beta: Inverse temperature parameter.
            allow_self_loops: Whether to allow self-loops in the generated graph.
        """

        self.embeddings = embeddings
        self.num_nodes = embeddings.shape[0]
        # initialize parameters
        self.alpha = alpha
        self.beta = beta
        self.allow_self_loops = allow_self_loops

        # intialize Hamiltonian entries for all node pairs
        self.H = self.hamiltonian()

        self.adj_matrix = None

    def energy_edge(self, i, j):
        """Compute energy for edge between nodes i and j."""
        x_i = self.embeddings[i]
        x_j = self.embeddings[j]

        # Compute Hamiltonian
        return utils.spring_constant(x_i, x_j, self.alpha) * utils.spring_energy(
            x_i, x_j
        )

    def hamiltonian(self):
        """Compute energy for directed edges between every node pair (Hamiltonian)."""
        H = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                H[i, j] = self.energy_edge(i, j)
        return H

    def generate(self, expected_num_edges):
        """
        Generate a graph based on the current node embeddings.

        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            Adjacency matrix of the generated graph.
        """
        generated_graph = np.zeros((self.num_nodes, self.num_nodes))
        c = self._compute_density_parameter(expected_num_edges)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if not self.allow_self_loops and i == j:
                    continue
                generated_graph[i, j] = self._generate_edges(i, j, c)

        self.adj_matrix = generated_graph
        return generated_graph

    def _compute_density_parameter(self, expected_num_edges):
        """
        Compute the density parameter c based on the expected number of edges in the generated graph.

        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            c: Density parameter.
        """
        if not self.allow_self_loops:
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


class SpringAttentionGenerativeModel:
    def __init__(self, X, Y, alpha=1, beta=1):
        """
        Initialize a generative SpringAttentionModel with the given parameters.

        Args:
            X: Node preference embeddings for the generative model.
            Y: Node status embeddings for the generative model.
            alpha: Variable spring constant weight.
            beta: Inverse temperature parameter.
        """

        assert X.shape == Y.shape, "X and Y must have the same shape."
        self.embeddings_x = X
        self.embeddings_y = Y
        self.num_nodes = X.shape[0]

        # initialize parameters
        self.alpha = alpha
        self.beta = beta

        # initialize unweighted hamiltonian
        self.H = self.hamiltonian()
        self.adj_matrix = None

    def energy_edge(self, i, j):
        """Compute energy for edge between nodes i and j."""
        x_i = self.embeddings_x[i]  # i preference
        y_i = self.embeddings_y[i]  # i status
        y_j = self.embeddings_y[j]  # j status

        # Compute Hamiltonian
        return utils.spring_constant(x_i, y_j) * utils.spring_energy(y_i, y_j)

    def hamiltonian(self):
        """Compute total energy of the system (Hamiltonian)."""
        total = 0.0
        for i, j in self.edge_list:
            total += self.energy_edge(i, j)
        return total

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

        self.adj_matrix = generated_graph
        return generated_graph

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


class SequentialHierarchyCommunitySimple(NetworkEmbeddingModel):
    def __init__(self, adj_matrix, embedding_dim, alpha=5, beta=1, k=None):
        """
        Initialize a model to learn community and hierarchy sequentially from a simple graph.

        Args:
            adj_matrix: Adjacency matrix of the input graph.
            embedding_dim: Dimension of the embedding space.
            beta: Inverse temperature parameter for the model.
            k: Power of the scaled cosine similarity that controls edge existence probability.
        """
        self.adj_matrix = adj_matrix
        self.embedding_dim = embedding_dim  # d
        if k is None:
            k = embedding_dim + 1
        self.k = k
        self.beta = beta
        self.alpha = alpha
        self.num_nodes = adj_matrix.shape[0]  # n
        self.embeddings = np.random.randn(
            self.num_nodes, self.embedding_dim
        )  # n x d matrix of embedding vectors

    def _edge_existence_prob(self, cos_sim_matrix, i, j):
        """
        Compute the probability of an edge existing between nodes i and j based on their embeddings.

        Args:
            s: Norm of the embedding vectors.
            d: cosine similarity between the embeddings of node i and j.
            i: Index of node i.
            j: Index of node j.

        Returns:
            Probability of an edge between nodes i and j.
        """
        return np.exp(self.alpha * (cos_sim_matrix[i, j] - 1))
    
    def _edge_direction_prob(self, s, cos_sim_matrix, i, j):
        """
        Compute the probability of an edge from node i to j conditioned on the edge existing.

        Args:
            s: Norm of the embedding vectors.
            d: Scaled cosine similarity matrix.
            i: Index of node i.
            j: Index of node j.

        Returns:
            Probability of an edge between nodes i and j.
        """
        sij = s[i] - s[j]
        cos_ij = cos_sim_matrix[i, j]
        return 1 / (1 + np.exp(-self.beta * (1 + cos_ij) * sij))

    def log_likelihood(self, X):
        s = np.linalg.norm(X, axis=1)
        cos_sim_matrix = utils.matrix_cosine_sim(X)
        L = 0.0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                L += self.adj_matrix[i, j] * np.log(self._edge_existence_prob(cos_sim_matrix, i, j) * self._edge_direction_prob(s, cos_sim_matrix, i, j) + 1e-9)
                L += (1 - self.adj_matrix[i, j]) * np.log(1 - self._edge_existence_prob(cos_sim_matrix, i, j) * self._edge_direction_prob(s, cos_sim_matrix, i, j) + 1e-9)
                # print((1 / alpha * (1 - np.exp(-2 * alpha))) * (1 + np.exp(-beta * sij * (1 + lij))) - np.exp(alpha * (lij - 1)) + 1e-9)
        return L

    def numerical_gradient(self, epsilon=1e-5):
        grad = np.zeros_like(self.embeddings)
        base = self.log_likelihood(self.embeddings)
        for i in range(self.num_nodes):
            for j in range(self.embedding_dim):
                X_perturb = self.embeddings.copy()
                X_perturb[i, j] += epsilon
                new_val = self.log_likelihood(X_perturb)
                grad[i, j] = (new_val - base) / epsilon
        return grad

    def optimize_embeddings(self, lr, max_iter=300, anneal=False):
        """
        Learn embeddings using annealing gradient descent.
        """
        history = []
        for it in tqdm(range(max_iter)):
            grad = self.numerical_gradient()
            adjusted_lr = lr / (1 + 0.05 * it) if anneal else lr
            self.embeddings += adjusted_lr * grad
            ll = self.log_likelihood(self.embeddings)
            history.append(ll)
        return (self.embeddings, history)

    def fit(self, lr=0.01, max_iter=300, anneal=False, plot_likelihood=True):
        """
        Fit the model to the data using annealing gradient descent.
        """
        self.embeddings, history = self.optimize_embeddings(lr, max_iter, anneal)

        if plot_likelihood:
            plt.plot(history)
            plt.title("Log Likelihood over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Log Likelihood")
            plt.grid(True)
            plt.show()

        return self.embeddings, history

    def predict(self, i, j):
        """
        Calculate the probability of a directed edge i -> j given the model's embeddings.
        """
        cos_sim_matrix = utils.matrix_cosine_sim(self.embeddings)
        s = np.linalg.norm(self.embeddings, axis=1)
        return self._edge_existence_prob(cos_sim_matrix, i, j) * self._edge_direction_prob(s, cos_sim_matrix, i, j)

    def generate(self, expected_num_edges=None):
        """
        Generate a synthetic network with the expected number of edges based on the learned embeddings.
        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            Adjacency matrix of the generated graph.
        """
        # Set the expected number of edges to the number of edges in the network
        if expected_num_edges is None:
            expected_num_edges = self.adj_matrix.sum()
        generated_adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        cos_sim_matrix = utils.matrix_cosine_sim(self.embeddings)
        c = expected_num_edges / np.sum(np.triu(cos_sim_matrix, 1))  # sum over dij for j > i
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                edge_prob = self.predict(i, j)
                generated_adj_matrix[i, j] = int(np.random.rand() < c * edge_prob)

        return generated_adj_matrix


class SequentialHierarchyCommunityMulti(NetworkEmbeddingModel):
    def __init__(self, adj_matrix, embedding_dim, beta=1, k=None):
        """
        Initialize a model to learn community and hierarchy sequentially from a simple graph.

        Args:
            adj_matrix: Adjacency matrix of the input graph.
            embedding_dim: Dimension of the embedding space.
            beta: Inverse temperature parameter for the model.
            k: Power of the scaled cosine similarity that controls edge existence probability.
        """
        self.adj_matrix = adj_matrix
        self.embedding_dim = embedding_dim  # d
        if k is None:
            k = embedding_dim + 1
        self.k = k
        self.beta = beta
        self.num_nodes = adj_matrix.shape[0]  # n
        self.embeddings = np.random.randn(
            self.num_nodes, self.embedding_dim
        )  # n x d matrix of embedding vectors

    def _directed_edge_cond_prob(self, s, d, i, j):
        """
        Compute the probability of an edge i -> j conditioned on the existence of an edge i <-> j.

        Args:
            s: Norm of the embedding vectors.
            d: Scaled cosine similarity matrix.
            i: Index of node i.
            j: Index of node j.

        Returns:
            Probability of an edge between nodes i and j.
        """
        sij = s[i] - s[j]
        dij = d[i, j]
        return 1 / (1 + np.exp(-2 * self.beta * sij * dij))

    def log_likelihood(self, X):
        s = np.linalg.norm(X, axis=1)
        d = utils.scaled_cosine_sim(X, self.k)
        A_bar = self.adj_matrix + self.adj_matrix.T
        m = np.sum(self.adj_matrix)
        D = np.sum(d) - np.sum(np.diag(d))
        L = 0.0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                pij = self._directed_edge_cond_prob(s, d, i, j)
                L += pij * np.log(self.adj_matrix[i, j] + 1e-9) + (1 - pij) * np.log(
                    self.adj_matrix[j, i] + 1e-9
                )  # edge direction terms
                L += (
                    A_bar[i, j] * np.log(d[i, j] / D + 1e-9) - (2 * m * d[i, j]) / D
                )  # edge existence terms
        return L

    def numerical_gradient(self, epsilon=1e-5):
        grad = np.zeros_like(self.embeddings)
        base = self.log_likelihood(self.embeddings)
        for i in range(self.num_nodes):
            for j in range(self.embedding_dim):
                X_perturb = self.embeddings.copy()
                X_perturb[i, j] += epsilon
                new_val = self.log_likelihood(X_perturb)
                grad[i, j] = (new_val - base) / epsilon
        return grad

    def optimize_embeddings(self, lr, max_iter=300, anneal=False):
        """
        Learn embeddings using annealing gradient descent.
        """
        history = []
        for it in tqdm(range(max_iter)):
            grad = self.numerical_gradient()
            alpha = lr / (1 + 0.05 * it) if anneal else lr
            self.embeddings += alpha * grad
            ll = self.log_likelihood(self.embeddings)
            history.append(ll)
        return (self.embeddings, history)

    def fit(self, lr=0.01, max_iter=300, anneal=False, plot_likelihood=True):
        """
        Fit the model to the data using annealing gradient descent.
        """
        self.embeddings, history = self.optimize_embeddings(lr, max_iter, anneal)

        if plot_likelihood:
            plt.plot(history)
            plt.title("Log Likelihood over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Log Likelihood")
            plt.grid(True)
            plt.show()

        return self.embeddings, history

    def predict_edge_count(self, i, j, c=None):
        """
        Predict the number of directed edges i -> j given the model's embeddings.
        Args:
            i: Index of node i.
            j: Index of node j.
            c: Density parameter for the model. If None, it will be estimated from the data.
        """
        d = utils.scaled_cosine_sim(self.embeddings, self.k)
        s = np.linalg.norm(self.embeddings, axis=1)
        if c is None:
            c_hat = np.sum(self.adj_matrix + self.adj_matrix.T) / (
                np.sum(d) - np.sum(np.diag(d))
            )
        else:
            c_hat = c
        edge_count = np.random.poisson(c_hat * d[i, j])
        return edge_count

    def predict(self, i, j):
        """
        Calculate the probability of a directed edge i -> j conditioned on the existence of an edge i <-> j.
        Args:
            i: Index of node i.
            j: Index of node j.
        """
        d = utils.scaled_cosine_sim(self.embeddings, self.k)
        s = np.linalg.norm(self.embeddings, axis=1)

        return self._directed_edge_cond_prob(s, d, i, j)

    def generate(self, expected_num_edges):
        """
        Generate a synthetic network with the expected number of edges based on the learned embeddings.
        Args:
            expected_num_edges: Expected number of edges in the generated graph.
        Returns:
            Adjacency matrix of the generated graph.
        """
        generated_adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        # compute c based on expected number of edges
        d = utils.scaled_cosine_sim(self.embeddings, self.k)
        c = expected_num_edges / np.sum(np.triu(d, 1))  # sum over dij for j > i
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # draw number of undirected edges between i and j
                edge_count = self.predict_edge_count(i, j, c)
                pij = self.predict(i, j)
                for _ in range(edge_count):
                    # pick edge direction for each generated edge
                    if np.random.rand() < pij:
                        generated_adj_matrix[i, j] += 1
                    else:
                        generated_adj_matrix[j, i] += 1

        return generated_adj_matrix
