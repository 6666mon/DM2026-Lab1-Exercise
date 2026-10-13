import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from langchain_core.tools import tool
import scipy.sparse as sp
from scipy.stats import rankdata

def make_tools(session):
    """
    Factory function that binds session state and returns feature filtering tools.
    """

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """
        Filters terms in the Document-Term Matrix based on feature variance.

        Calculates the variance of word counts for each term across all documents.
        Uses sparse matrix operations (E[X^2] - (E[X])^2) to compute variance efficiently
        without densifying large matrices. Terms with variance strictly less than the
        specified threshold are flagged for removal.

        Args:
            threshold: Minimum variance required to keep a term. Default is 0.0.

        Returns:
            A summary dictionary containing result_id, threshold, total_terms,
            retained_terms_count, removed_terms_count, retained_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found in session. Run build_dtm_tool first.")

        matrix = session.feature_matrix

        # Compute per-term variance in a memory-efficient sparse manner using Var(X) = E[X^2] - (E[X])^2
        if sp.issparse(matrix):
            mean = np.array(matrix.mean(axis=0)).ravel()
            mean_sq = np.array(matrix.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
            # Clip small floating-point rounding errors near 0.0
            variances = np.maximum(variances, 0.0)
        else:
            dense_matrix = np.asarray(matrix)
            variances = np.var(dense_matrix, axis=0, ddof=0)

        # Build full report DataFrame for visualize_result_tool
        # Rule: Column 1 = term name, Column 2 = metric value (variance), sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Filter terms: terms with variance < threshold are removed
        retained_mask = variances >= threshold
        retained_terms = [name for name, keep in zip(session.feature_names, retained_mask) if keep]
        removed_terms = [name for name, keep in zip(session.feature_names, retained_mask) if not keep]

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "retained_terms_count": len(retained_terms),
            "removed_terms_count": len(removed_terms),
            "retained_terms": retained_terms,
            "removed_terms": removed_terms
        }

        # Store result with full_report formatted for visualize_result_tool
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        abs_value: bool = False
    ) -> Dict[str, Any]:
        """
        Filters terms based on Pearson correlation between word counts and a target class.

        Computes linear correlation r between each term's count vector and a binary indicator
        vector for target_class (one-vs-rest). Handles constant/zero-variance columns by setting
        correlation to 0.0.

        Args:
            target_class: Target category label to correlate terms against (e.g. "catB").
            threshold: Minimum Pearson correlation coefficient to retain a term. Default is 0.0.
            abs_value: If True, filters based on absolute correlation |r| >= threshold. Default is False.

        Returns:
            A summary dictionary containing result_id, target_class, threshold, abs_value,
            total_terms, retained_terms_count, removed_terms_count, retained_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found in session. Run build_dtm_tool first.")

        # Determine target labels
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values
        else:
            raise ValueError("No category labels found in session.")

        # Create binary indicator vector for target_class (one-vs-rest)
        y = (np.array(labels) == target_class).astype(float)
        if np.sum(y) == 0:
            raise ValueError(f"Target class '{target_class}' not found in category labels.")

        matrix = session.feature_matrix
        if sp.issparse(matrix):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        # Compute Pearson correlation r per term
        # r = Cov(X, Y) / (std(X) * std(Y))
        y_diff = y - np.mean(y)
        y_std = np.std(y)

        pearson_rs = []
        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]
            x_diff = x - np.mean(x)
            x_std = np.std(x)

            if x_std == 0 or y_std == 0:
                r = 0.0
            else:
                r = np.mean(x_diff * y_diff) / (x_std * y_std)
                # Handle possible floating point precision clipping to [-1.0, 1.0]
                r = float(np.clip(r, -1.0, 1.0))

            if np.isnan(r):
                r = 0.0

            pearson_rs.append(r)

        pearson_rs = np.array(pearson_rs)

        # Build full report DataFrame for visualize_result_tool
        # Rule: Column 1 = term, Column 2 = pearson_r, sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": pearson_rs
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Filter terms
        if abs_value:
            retained_mask = np.abs(pearson_rs) >= threshold
        else:
            retained_mask = pearson_rs >= threshold

        retained_terms = [name for name, keep in zip(session.feature_names, retained_mask) if keep]
        removed_terms = [name for name, keep in zip(session.feature_names, retained_mask) if not keep]

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "abs_value": abs_value,
            "total_terms": len(session.feature_names),
            "retained_terms_count": len(retained_terms),
            "removed_terms_count": len(removed_terms),
            "retained_terms": retained_terms,
            "removed_terms": removed_terms
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold, "abs_value": abs_value},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        abs_value: bool = False
    ) -> Dict[str, Any]:
        """
        Filters terms based on Spearman rank correlation between word counts and a target class.

        Computes monotonic rank correlation (rho) between each term's count vector and a binary
        indicator vector for target_class (one-vs-rest). Rank transforms both inputs prior to
        correlation. Handles constant/zero-variance columns by setting correlation to 0.0.

        Args:
            target_class: Target category label to correlate terms against (e.g. "catB").
            threshold: Minimum Spearman correlation coefficient to retain a term. Default is 0.0.
            abs_value: If True, filters based on absolute correlation |rho| >= threshold. Default is False.

        Returns:
            A summary dictionary containing result_id, target_class, threshold, abs_value,
            total_terms, retained_terms_count, removed_terms_count, retained_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found in session. Run build_dtm_tool first.")

        # Determine target labels
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values
        else:
            raise ValueError("No category labels found in session.")

        # Create binary indicator vector for target_class (one-vs-rest)
        y = (np.array(labels) == target_class).astype(float)
        if np.sum(y) == 0:
            raise ValueError(f"Target class '{target_class}' not found in category labels.")

        matrix = session.feature_matrix
        if sp.issparse(matrix):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        # Compute Spearman rank correlation rho per term
        # rho = Pearson(rank(X), rank(Y))
        y_rank = rankdata(y, method="average")
        y_diff = y_rank - np.mean(y_rank)
        y_std = np.std(y_rank)

        spearman_rhos = []
        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]
            x_rank = rankdata(x, method="average")
            x_diff = x_rank - np.mean(x_rank)
            x_std = np.std(x_rank)

            if x_std == 0 or y_std == 0:
                rho = 0.0
            else:
                rho = np.mean(x_diff * y_diff) / (x_std * y_std)
                rho = float(np.clip(rho, -1.0, 1.0))

            if np.isnan(rho):
                rho = 0.0

            spearman_rhos.append(rho)

        spearman_rhos = np.array(spearman_rhos)

        # Build full report DataFrame for visualize_result_tool
        # Rule: Column 1 = term, Column 2 = spearman_r, sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": spearman_rhos
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Filter terms
        if abs_value:
            retained_mask = np.abs(spearman_rhos) >= threshold
        else:
            retained_mask = spearman_rhos >= threshold

        retained_terms = [name for name, keep in zip(session.feature_names, retained_mask) if keep]
        removed_terms = [name for name, keep in zip(session.feature_names, retained_mask) if not keep]

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "abs_value": abs_value,
            "total_terms": len(session.feature_names),
            "retained_terms_count": len(retained_terms),
            "removed_terms_count": len(removed_terms),
            "retained_terms": retained_terms,
            "removed_terms": removed_terms
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold, "abs_value": abs_value},
            summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
