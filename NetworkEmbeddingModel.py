import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, kendalltau

import utils


class NetworkEmbeddingModel:

    def get_embedding_norms(self, embeddings=None):
        """Return the magnitudes of node embeddings (continuous ranks)."""
        if embeddings is None:
            embeddings = self.embeddings
        return np.linalg.norm(embeddings, axis=1)

    def get_community_structure(
        self, embeddings=None, method="agglomerative", n_clusters=None
    ):
        """
        Extract community structure from embeddings using either agglomerative or spectral clustering

        Parameters:
        -----------
        embeddings : numpy.ndarray or None
            Embeddings to use. If None, use current embeddings.
        method : str
            Clustering method ('agglomerative' or 'spectral')
        n_clusters : int or None
            Number of communities to detect

        Returns:
        --------
        communities : list
            Node assignments to communities
        """

        if embeddings is None:
            embeddings = self.embeddings

        if n_clusters is None:
            # Estimate number of clusters using silhouette score
            sil_scores = []
            max_clusters = min(20, self.num_nodes // 5 + 1)  # Reasonable upper bound
            range_n_clusters = range(2, max_clusters)

            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                if len(set(cluster_labels)) > 1:  # Check if we have at least 2 clusters
                    silhouette_avg = silhouette_score(embeddings, cluster_labels)
                    sil_scores.append(silhouette_avg)
                else:
                    sil_scores.append(-1)

            n_clusters = range_n_clusters[np.argmax(sil_scores)]
            print(f"Estimated optimal number of communities: {n_clusters}")

        if method == "agglomerative":
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
        elif method == "spectral":
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="cosine", random_state=42
            )
        else:
            raise ValueError("Unsupported clustering method")

        return clustering.fit_predict(embeddings)

    def evaluate_clusters(
        self,
        true_communities,
        metric="nmi",
        embeddings=None,
        clustering_method="agglomerative",
        n_clusters=None,
    ):
        """
        Evaluate the community structure using NMI or the silhouette score.

        Parameters:
        -----------
        true_communities : array-like
            Ground truth community assignments for nodes
        metric : str
            Clustering evaluation metric ('nmi' or 'silhouette')
        embeddings : numpy.ndarray or None
            Embeddings to use. If None, use current embeddings.
        clustering_method : str
            Clustering method ('agglomerative' or 'spectral')
        n_clusters : int or None
            Number of communities to detect

        Returns:
        --------
        score : float, community_ranks
            Score of the clustering based on the specified metric and community ranks calculated as average magnitude of embeddings for each community
        """
        assert metric in [
            "nmi",
            "silhouette",
        ], "metric must be either 'nmi' or 'silhouette'"
        assert clustering_method in [
            "agglomerative",
            "spectral",
        ], "clustering_method must be either 'agglomerative' or 'spectral'"
        if embeddings is None:
            embeddings = self.embeddings

        communities = self.get_community_structure(
            embeddings=embeddings, method=clustering_method, n_clusters=n_clusters
        )

        if metric == "nmi":
            return normalized_mutual_info_score(
                true_communities, communities, average_method="arithmetic"
            )
        elif metric == "silhouette":
            # Compute silhouette score
            # Note: silhouette_score expects a distance matrix, so we need to convert the adjacency matrix to a distance matrix
            # Here we use 1 - cosine similarity as the distance metric
            dist_matrix = 1 - utils.matrix_cosine_sim(embeddings)
            dist_matrix[dist_matrix < 0] = 0
            return silhouette_score(dist_matrix, communities)

    def evaluate_community_ranks(
        self,
        true_community_ranks,
        communities,
        metric="spearman",
        embeddings=None,
    ):
        """
        Evaluate the ranks of groups average across their constituent nodes using Spearman or Kendall correlation. The true community ranks must match the number of communities provided.
        """
        assert metric in [
            "spearman",
            "kendall",
        ], "metric must be either 'spearman' or 'kendall'"
        assert len(true_community_ranks) == len(
            set(communities)
        ), "true_community_ranks and communities must have the same length"

        if embeddings is None:
            embeddings = self.embeddings

        norms = self.get_embedding_norms(embeddings)
        n_communities = communities.max() + 1
        community_ranks = np.zeros(n_communities)
        for community_label in range(n_communities):
            community_ranks[community_label] = np.mean(
                norms[communities == community_label]
            )
        # Compute the rank correlation between true community ranks and predicted community ranks
        if metric == "spearman":
            return spearmanr(true_community_ranks, community_ranks)[0], community_ranks
        elif metric == "kendall":
            return kendalltau(true_community_ranks, community_ranks)[0], community_ranks

    def evaluate_ranks_within_communities(
        self,
        true_ranks,
        communities,
        metric="spearman",
        embeddings=None,
    ):
        """
        Evaluate the ranks of nodes within each group using Spearman or Kendall correlation. The within-group true ranks must match the community assignments provided.

        Parameters:
        -----------
        true_ranks : array-like
            Ground truth within-community ranks for nodes
        communities : array-like
            Community assignments for nodes
        metric : str
            Rank evaluation metric ('spearman' or 'kendall')
        embeddings : numpy.ndarray or None
            Embeddings to use. If None, use current embeddings.

        Returns:
        --------
        score : float
            Score of the rank evaluation based on the specified metric
        """
        assert metric in [
            "spearman",
            "kendall",
        ], "metric must be either 'spearman' or 'kendall'"
        assert len(true_ranks) == len(
            communities
        ), "true_ranks and communities must have the same length"

        if embeddings is None:
            embeddings = self.embeddings

        norms = self.get_embedding_norms(embeddings)
        n_communities = communities.max() + 1  # assumes communities are 0-indexed

        within_community_correlations = []
        for community_label in range(n_communities):
            true_within_community_ranks = np.array(true_ranks)[
                communities == community_label
            ]
            within_community_ranks = np.array(norms)[communities == community_label]
            correlation = self.evaluate_ranks(
                true_within_community_ranks, within_community_ranks, metric=metric
            )
            within_community_correlations.append(correlation)

        return within_community_correlations

    def evaluate_ranks(self, true_ranks, pred_ranks, metric="spearman"):
        """
        Evaluate the ranks of nodes using Spearman or Kendall correlation.

        Parameters:
        -----------
        true_ranks : array-like
            Ground truth ranks for nodes
        pred_ranks : array-like
            Predicted ranks for nodes
        metric : str
            Rank evaluation metric ('spearman' or 'kendall')

        Returns:
        --------
        score : float
            Score of the rank evaluation based on the specified metric
        """
        assert metric in [
            "spearman",
            "kendall",
        ], "metric must be either 'spearman' or 'kendall'"
        assert len(true_ranks) == len(
            pred_ranks
        ), "true_ranks and pred_ranks must have the same length"

        if metric == "spearman":
            return spearmanr(true_ranks, pred_ranks)[0]
        elif metric == "kendall":
            return kendalltau(true_ranks, pred_ranks)[0]

    def visualize(
        self,
        ax=None,
        embeddings=None,
        node_labels=None,
        draw_labels=True,
        draw_edges=True,
        draw_legend=True,
        adj_matrix=None,
        communities=None,
        community_names=None,
        show_magnitudes=True,
        node_size_base=100,
        figsize=(20, 16),
        title=None,
        dim_reduction="pca",
    ):
        """
        Visualize the network with embeddings.

        Parameters:
        -----------
        embeddings : numpy.ndarray or None
            Embeddings to visualize. If None, use current embeddings.
        communities : array-like or None
            Community assignments for nodes
        show_magnitudes : bool
            Whether to show node magnitudes as node sizes
        node_size_base : int
            Base size for nodes in visualization
        figsize : tuple
            Figure size
        """
        # require both embeddings and adjacency matrix for visualization
        assert (
            embeddings is not None or self.embeddings is not None
        ), "Either embeddings or self.embeddings must be provided for visualization"
        assert (
            adj_matrix is not None or self.adj_matrix is not None
        ), "Either adj_matrix or self.adj_matrix must be provided for visualization"
        assert dim_reduction in [
            "pca",
            "tsne",
        ], "dim_reduction must be either 'pca' or 'tsne'"
        if embeddings is None:
            embeddings = self.embeddings

        if adj_matrix is None:
            adj_matrix = self.adj_matrix

        # Create graph from adjacency matrix
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        if node_labels is not None:
            # Use provided node labels
            assert (
                len(node_labels) == self.num_nodes
            ), "Node labels must match number of nodes"
            G = nx.relabel_nodes(G, {i: node_labels[i] for i in range(self.num_nodes)})
        else:
            node_labels = list(G.nodes())
        # Node sizes based on magnitudes (ranks)
        if show_magnitudes:
            magnitudes = self.get_embedding_norms(embeddings)
            node_sizes = (
                magnitudes / max(magnitudes) * node_size_base * 3 + node_size_base
            )
        else:
            node_sizes = node_size_base

        # Get positions from embeddings
        if embeddings.shape[1] > 2:
            # Reduce to 2D using t-SNE
            if dim_reduction == "tsne":
                embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(
                    embeddings
                )
            else:
                # reduce using PCA
                embeddings_2d = PCA(n_components=2, random_state=42).fit_transform(
                    embeddings
                )
        else:
            embeddings_2d = embeddings[:, :2]

        pos = {
            node_labels[i]: (embeddings_2d[i, 0], embeddings_2d[i, 1])
            for i in range(len(node_labels))
        }

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if communities is not None:
            # Color nodes by community
            cmap = plt.cm.Pastel1.colors
            community_colors = [cmap[i] for i in communities]
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=community_colors,
                # cmap=plt.cm.tab20.colors,
                node_size=node_sizes,
                alpha=0.8,
            )
            if draw_legend:
                legend_elements = [
                    Patch(
                        facecolor=cmap[i],
                        label=(
                            f"Community {i}"
                            if community_names is None
                            else community_names[i]
                        ),
                    )
                    for i in sorted(set(communities))
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="lower right",
                    fontsize=8,
                    bbox_to_anchor=(1.25, 0.0),
                )
        else:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, alpha=0.8)

        if draw_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                arrowsize=10,
                connectionstyle="arc3,rad=0.1",
                edge_color="0.5",
                width=0.5,
            )

        if draw_labels:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        # hide all axes
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

        ax.set_title("Network embedding visualization" if title is None else title)
        ax.axis("off")
