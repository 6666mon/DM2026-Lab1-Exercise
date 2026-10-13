import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    """
    Factory function that binds session state and returns tools for DTM operations.
    """

    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        min_df: float = 1,
        max_df: float = 1.0,
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) from text data in session.dataframe.

        Converts a collection of text documents to a sparse matrix of token counts
        using CountVectorizer. Calculates matrix sparsity and stores vectorizer artifacts
        on the shared session state for downstream analysis.

        Args:
            text_column: Name of the column in session.dataframe containing raw text.
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams (e.g. [1, 1]).
            min_df: Minimum document frequency threshold (count or fraction).
            max_df: Maximum document frequency threshold (count or fraction).
            stop_words: Language stop words to exclude (e.g. "english" or None).

        Returns:
            A summary dictionary containing result_id, n_documents, n_terms, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session dataframe.")

        # Convert list to tuple to satisfy scikit-learn CountVectorizer parameter requirement
        ngram_tuple = (ngram_range[0], ngram_range[1])

        # Handle NaNs or missing text values by converting to empty string
        texts = session.dataframe[text_column].fillna("").astype(str).tolist()

        # Instantiate and fit CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words
        )
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Sync labels if category info is present
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Compute sparsity metrics
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 10) -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents in the Document-Term Matrix.

        Calculates the total occurrence count for each term across all documents in
        session.feature_matrix. Sorts terms by total frequency descending and reports
        top terms summary and total corpus word count.

        Args:
            top_n: Number of top frequent terms to return in the summary. Default is 10.

        Returns:
            A summary dictionary containing result_id, total_terms, total_word_count,
            and top_terms (dict mapping term name to total frequency).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found in session. Run build_dtm_tool first.")

        matrix = session.feature_matrix

        # Sum occurrences per column (term) across all documents (rows)
        counts = np.array(matrix.sum(axis=0)).ravel()

        # Build full report DataFrame for visualize_result_tool
        # Rule: Column 1 = term, Column 2 = frequency, sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Extract top_n terms as a dictionary mapping term -> int frequency
        df_top = df_report.head(top_n)
        top_terms_dict = {row["term"]: int(row["frequency"]) for _, row in df_top.iterrows()}

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "total_word_count": int(np.sum(counts)),
            "top_terms": top_terms_dict
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """
        Generates a heatmap visualization of a raw positional slice of the Document-Term Matrix.

        Slices the first n_documents (rows) and first n_terms (columns) from session.feature_matrix
        in their existing positional order (no sorting by variance or frequency). Produces a seaborn
        heatmap and attaches the figure to session.pending_figure.

        Args:
            n_terms: Maximum number of terms (columns) to include from the start of the vocabulary. Default is 20.
            n_documents: Maximum number of documents (rows) to include from the start of the dataset. Default is 20.

        Returns:
            A summary dictionary containing result_id, n_documents, n_terms, terms, document_ids,
            and matrix (a nested list representing the sliced count matrix).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found in session. Run build_dtm_tool first.")

        # Determine effective dimensions based on requested bounds vs matrix size
        total_docs, total_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        # Slice the positional matrix and terms
        sliced_matrix = session.feature_matrix[0:actual_docs, 0:actual_terms]
        dense_slice = sliced_matrix.toarray() if hasattr(sliced_matrix, "toarray") else np.asarray(sliced_matrix)
        sliced_terms = session.feature_names[0:actual_terms]
        doc_labels = [f"Doc {i}" for i in range(actual_docs)]

        # Generate figure and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (First {actual_docs} Docs x First {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_terms,
            "document_ids": doc_labels,
            "matrix": dense_slice.tolist()
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
