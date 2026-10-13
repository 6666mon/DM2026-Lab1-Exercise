"""Dimensionality reduction tools for the Data Mining lab pipeline.

Provides tools for reducing high-dimensional document-term matrices to 2D coordinates
using PCA, t-SNE, or UMAP, and visualizing the reduced representation, as well as
binarizing category labels.
"""

from typing import Dict, Any, Optional, Literal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning all reduction tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensionality using PCA, t-SNE, or UMAP and plots 2D coordinates.

        Requires session.feature_matrix to be populated (e.g. via build_dtm_tool).
        Computes 2D reduced representation, generates a scatter plot colored by category on
        session.pending_figure, stores coordinates in session.artifacts, and returns a summary.

        Args:
            method: Reduction method to use ("pca", "tsne", or "umap"). Default is "pca".
            n_components: Number of dimensions to reduce to (typically 2). Default is 2.
            perplexity: Perplexity parameter for t-SNE (used when method="tsne").
            n_neighbors: Number of neighbors parameter for UMAP (used when method="umap").
            random_state: Random seed for reproducibility (t-SNE/UMAP or randomized PCA).

        Returns:
            Dict containing result_id, method, reduced coordinates, and explained variance (if PCA).
        """
        if session.feature_matrix is None:
            return {
                "error": "No feature_matrix found in session. Please run build_dtm_tool first."
            }

        X = session.feature_matrix.toarray()
        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            X_reduced = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="random",
                learning_rate=200.0,
            )
            X_reduced = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            X_reduced = reducer.fit_transform(X)
        else:
            return {"error": f"Unsupported reduction method: '{method}'."}

        # Store reduced coordinates in session.artifacts
        session.artifacts["reduced_coordinates"] = X_reduced
        session.artifacts[f"reduced_{method}"] = X_reduced

        # Generate mandatory scatter plot colored by document category
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            categories = list(session.labels)

        if categories is not None:
            unique_cats = sorted(list(set(categories)))
            for cat in unique_cats:
                indices = [i for i, c in enumerate(categories) if c == cat]
                ax.scatter(
                    X_reduced[indices, 0],
                    X_reduced[indices, 1],
                    label=str(cat),
                    alpha=0.8,
                    edgecolors="none",
                )
            ax.legend(title="Category")
        else:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.8)

        dim_label_1 = "PC1" if method == "pca" else "Dim 1"
        dim_label_2 = "PC2" if method == "pca" else "Dim 2"
        ax.set_xlabel(dim_label_1)
        ax.set_ylabel(dim_label_2)
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": X_reduced.tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {"method": method}, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes document category labels into a one-hot indicator matrix.

        Reads labels from session.dataframe['category_name'] or session.labels,
        applies sklearn LabelBinarizer, stores the binary matrix in session.artifacts,
        and returns a summary of classes and shape.

        Returns:
            Dict containing result_id, classes, shape (n_samples, n_classes),
            and binarized_labels matrix.
        """
        categories = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            categories = list(session.labels)

        if categories is None or len(categories) == 0:
            return {
                "error": "No category labels found in session. Please load a dataset first."
            }

        lb = LabelBinarizer()
        binary_matrix = lb.fit_transform(categories)

        # Handle binary classification case where LabelBinarizer outputs a 1D column (N, 1)
        # by converting it to 2D 2-column one-hot encoding if there are 2 classes
        classes = lb.classes_.tolist()
        if len(classes) == 2 and binary_matrix.shape[1] == 1:
            binary_matrix = np.hstack([1 - binary_matrix, binary_matrix])

        session.artifacts["binarized_labels"] = binary_matrix
        session.artifacts["binarizer_classes"] = classes

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": list(binary_matrix.shape),
            "binarized_labels": binary_matrix.tolist(),
        }

        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
