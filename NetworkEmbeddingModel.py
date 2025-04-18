import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


class NetworkEmbeddingModel:

    def get_node_ranks(self, embeddings=None):
        """Return the magnitudes of node embeddings (continuous ranks)."""
        if embeddings is None:
            embeddings = self.embeddings
        return np.linalg.norm(embeddings, axis=1)

    def get_community_structure(
        self, embeddings=None, method="kmeans", n_clusters=None
    ):
        """
        Extract community structure from embeddings using either k-means or spectral clustering.

        Parameters:
        -----------
        embeddings : numpy.ndarray or None
            Embeddings to use. If None, use current embeddings.
        method : str
            Clustering method ('kmeans' or 'spectral')
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

        if method == "kmeans":
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "spectral":
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42
            )
        else:
            raise ValueError("Unsupported clustering method")

        return clustering.fit_predict(embeddings)

    def visualize(
        self,
        embeddings=None,
        adj_matrix=None,
        communities=None,
        show_magnitudes=True,
        node_size_base=100,
        figsize=(12, 10),
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
        if embeddings is None:
            embeddings = self.embeddings

        if adj_matrix is None:
            adj_matrix = self.adj_matrix

        # Create graph from adjacency matrix
        G = nx.from_numpy_array(adj_matrix)

        # Node sizes based on magnitudes (ranks)
        if show_magnitudes:
            magnitudes = self.get_node_ranks(embeddings)
            node_sizes = (
                magnitudes / max(magnitudes) * node_size_base * 3 + node_size_base
            )
        else:
            node_sizes = node_size_base

        # Get positions from embeddings
        if embeddings.shape[1] > 2:
            # Reduce to 2D using t-SNE
            embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(
                embeddings
            )
        else:
            embeddings_2d = embeddings[:, :2]

        pos = {
            i: (embeddings_2d[i, 0], embeddings_2d[i, 1]) for i in range(self.num_nodes)
        }

        # Plot
        fig, _ = plt.subplots(figsize=figsize)

        if communities is not None:
            # Color nodes by community
            cmap = plt.cm.tab20.colors
            community_colors = [cmap[i] for i in communities]
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color=community_colors,
                # cmap=plt.cm.tab20.colors,
                node_size=node_sizes,
                alpha=0.8,
            )
            legend_elements = [
                Patch(facecolor=cmap[i], label=f"Community {i}")
                for i in sorted(set(communities))
            ]
            plt.legend(handles=legend_elements)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8)

        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # hide all axes
        plt.gca().spines[["top", "right"]].set_visible(False)
        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(0, color="black", linewidth=1)

        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        plt.title("Network embedding visualization")
        plt.axis("off")
        plt.show()
