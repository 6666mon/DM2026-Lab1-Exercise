"""Exploration tools for corpus feature and document similarity analysis."""

from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session: Any) -> List[Any]:
    """Factory function creating exploration tools bound to the session state."""

    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> Dict[str, Any]:
        """Compute the cosine similarity between two documents in the document-term matrix.

        Args:
            doc_index1: 0-based index of the first document.
            doc_index2: 0-based index of the second document.

        Returns:
            Dict containing the result_id, doc indices, similarity score, and category labels.
        """
        if session.feature_matrix is None:
            return {
                "error": "Document-term matrix has not been built yet. Please run build_dtm_tool first."
            }

        n_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_index1 < n_docs) or not (0 <= doc_index2 < n_docs):
            return {
                "error": f"Document index out of bounds. Valid indices are 0 to {n_docs - 1}."
            }

        vec1 = session.feature_matrix[doc_index1]
        vec2 = session.feature_matrix[doc_index2]

        norm1 = np.linalg.norm(vec1.toarray()) if hasattr(vec1, "toarray") else np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2.toarray()) if hasattr(vec2, "toarray") else np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            sim_score = 0.0
        else:
            sim_score = float(cosine_similarity(vec1, vec2)[0][0])

        result_id = session.next_result_id("cosine_similarity")

        label1 = None
        label2 = None
        if session.labels is not None:
            label1 = str(session.labels[doc_index1])
            label2 = str(session.labels[doc_index2])

        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "category1": label1,
            "category2": label2,
            "cosine_similarity": sim_score,
        }

        session.store_result("cosine_similarity_tool", {"doc_index1": doc_index1, "doc_index2": doc_index2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Compute the Pearson feature correlation matrix over the top terms by variance.

        Args:
            top_n: Maximum number of top terms by variance to include (default 20).

        Returns:
            Dict containing result_id, terms selected, and the correlation matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-term matrix has not been built yet. Please run build_dtm_tool first."
            }

        dense_matrix = (
            session.feature_matrix.toarray()
            if hasattr(session.feature_matrix, "toarray")
            else np.asarray(session.feature_matrix)
        )

        n_docs, n_features = dense_matrix.shape
        if n_features == 0:
            return {"error": "Document-term matrix has no features."}

        # Compute variance per term across documents
        variances = np.var(dense_matrix, axis=0)

        # Select top N terms by variance
        k = min(top_n, n_features)
        top_indices = np.argsort(variances)[::-1][:k]

        top_terms = [session.feature_names[i] for i in top_indices]
        top_matrix = dense_matrix[:, top_indices]

        # Compute Pearson correlation matrix across top terms (terms as variables)
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Handle case where k=1 (np.corrcoef returns scalar)
        if k == 1:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(max(6, k * 0.4), max(5, k * 0.4)))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Heatmap (Top {k} Terms by Variance)")
        plt.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        corr_matrix_list = [[float(val) for val in row] for row in corr_matrix]

        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_matrix_list,
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
