import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Union, List
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function that binds session state and returns tools for dataset exploration and preprocessing.
    """

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """
        Lists files and directories within a workspace subdirectory.

        Enables discovery of dataset files and directory structure prior to loading.

        Args:
            subdirectory: Relative path within the workspace root to list. Default is ".".

        Returns:
            A summary dictionary containing result_id, subdirectory, directories list, and files list.
        """
        target_dir = os.path.normpath(subdirectory)
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory '{subdirectory}' does not exist.")

        entries = os.listdir(target_dir)
        dirs = sorted([e for e in entries if os.path.isdir(os.path.join(target_dir, e))])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(target_dir, e))])

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": dirs,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """
        Inspects the top rows of the loaded dataset in session.dataframe.

        Args:
            n_rows: Number of initial rows to return from session.dataframe. Default is 5.

        Returns:
            A summary dictionary containing result_id, n_rows, total_rows, columns, and sample_records.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        actual_n = min(n_rows, 5) if n_rows > 0 else 5
        df = session.dataframe
        df_head = session.dataframe.head(actual_n)

        records = df_head.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, str) and len(v) > 200:
                    rec[k] = v[:200] + "..."

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_rows": len(df_head),
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_records": records
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    @tool
    def check_missing_tool(column: str = "text") -> Dict[str, Any]:
        """
        Checks for missing or empty string values in a column of session.dataframe.

        Args:
            column: Name of the column to check for missing/empty values. Default is "text".

        Returns:
            A summary dictionary containing result_id, column, total_rows, missing_count, missing_pct, and missing_indices.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if column not in session.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in session.dataframe.")

        series = session.dataframe[column]
        is_missing = series.isna() | (series.astype(str).str.strip() == "")
        missing_indices = series[is_missing].index.tolist()
        missing_count = len(missing_indices)
        total_rows = len(session.dataframe)
        missing_pct = float(100.0 * missing_count / total_rows) if total_rows > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "column": column,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {"column": column}, summary)
        return summary

    @tool
    def check_duplicates_tool(subset: Optional[Union[str, List[str]]] = "text", drop: bool = False) -> Dict[str, Any]:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.

        When drop=True, uses keep=False (drops all copies of duplicate rows).

        Args:
            subset: Column name or list of column names to check for duplicates. Default is "text".
            drop: Whether to drop duplicate rows from session.dataframe. Default is False.

        Returns:
            A summary dictionary containing result_id, total_rows, duplicate_count, duplicate_pct, dropped, and remaining_rows.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if isinstance(subset, str):
            check_sub = [subset]
        elif subset is None:
            check_sub = None
        else:
            check_sub = subset

        if check_sub is not None:
            for col in check_sub:
                if col not in session.dataframe.columns:
                    raise ValueError(f"Column '{col}' not found in session.dataframe.")

        total_rows = len(session.dataframe)
        is_dup = session.dataframe.duplicated(subset=check_sub, keep=False)
        duplicate_count = int(is_dup.sum())
        duplicate_pct = float(100.0 * duplicate_count / total_rows) if total_rows > 0 else 0.0

        if drop and duplicate_count > 0:
            session.dataframe = session.dataframe[~is_dup].reset_index(drop=True)

        remaining_rows = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "subset": subset,
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "duplicate_pct": round(duplicate_pct, 4),
            "dropped": drop,
            "remaining_rows": remaining_rows
        }

        session.store_result("check_duplicates_tool", {"subset": subset, "drop": drop}, summary)
        return summary

    @tool
    def tokenize_tool(column: str = "text") -> Dict[str, Any]:
        """
        Tokenizes text in session.dataframe into unigrams using nltk.word_tokenize.

        Standardizes document text into lowercased alphanumeric unigrams and stores
        the resulting list of tokens in session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Args:
            column: Column name in session.dataframe containing document text. Default is "text".

        Returns:
            A summary dictionary containing result_id, total_documents, total_tokens, avg_tokens_per_doc, and vocabulary_size.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if column not in session.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in session.dataframe.")

        # Ensure NLTK tokenizer resources are available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        tokenized_docs = []
        all_tokens = []

        for val in session.dataframe[column]:
            text_str = str(val) if pd.notna(val) else ""
            if not text_str.strip():
                tokens = []
            else:
                raw_tokens = nltk.word_tokenize(text_str.lower())
                tokens = [t for t in raw_tokens if t.isalnum()]

            tokenized_docs.append(tokens)
            all_tokens.extend(tokens)

        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        total_docs = len(session.dataframe)
        total_tokens = len(all_tokens)
        avg_tokens = round(float(total_tokens / total_docs), 4) if total_docs > 0 else 0.0
        vocab_size = len(set(all_tokens))

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "column": column,
            "total_documents": total_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "vocabulary_size": vocab_size
        }

        session.store_result("tokenize_tool", {"column": column}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False
    ) -> Dict[str, Any]:
        """
        Subsamples session.dataframe using pandas DataFrame.sample.

        Updates session.dataframe in place with the sampled subset.

        Args:
            n: Number of items from axis to return. Cannot be used with frac.
            frac: Fraction of axis items to return. Cannot be used with n.
            random_state: Seed for the random number generator for reproducible sampling.
            replace: Allow or disallow sampling of the same row more than once. Default is False.

        Returns:
            A summary dictionary containing result_id, original_rows, sampled_rows, sample_indices, and sample_records.
            Text values in sample_records are truncated to at most 200 characters, sample_records contains at most 5 items,
            and sample_indices contains at most 50 indices, while sampled_rows reports the exact size of the sampled dataset.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if n is None and frac is None:
            n = 5  # Default sample size if neither n nor frac is provided

        original_rows = len(session.dataframe)

        sampled_df = session.dataframe.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state
        )

        sample_indices = sampled_df.index.tolist()
        sampled_rows = len(sampled_df)

        # Update session.dataframe in place and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        # Format sample records for display: slice to at most 5 records and truncate text strings to 200 chars
        display_records = session.dataframe.head(5).to_dict(orient="records")
        for rec in display_records:
            for k, v in rec.items():
                if isinstance(v, str) and len(v) > 200:
                    rec[k] = v[:200]

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "original_rows": original_rows,
            "sampled_rows": sampled_rows,
            "sample_indices": sample_indices[:50],
            "sample_records": display_records
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state, "replace": replace},
            summary
        )
        return summary

    @tool
    def describe_data_tool(text_column: str = "text") -> Dict[str, Any]:
        """
        Computes summary statistics on text length and builds a box plot by category.

        Computes character counts for text_column in session.dataframe, calculates overall
        and per-category summary statistics (.describe()), and sets a box plot figure on
        session.pending_figure.

        Args:
            text_column: Name of the column in session.dataframe containing text. Default is "text".

        Returns:
            A summary dictionary containing result_id, overall_stats, and by_category stats.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        df = session.dataframe.copy()
        df["text_length"] = df[text_column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        overall_desc = df["text_length"].describe().to_dict()
        overall_stats = {k: round(float(v), 4) for k, v in overall_desc.items()}

        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        by_category = {}
        if category_col:
            grouped = df.groupby(category_col)["text_length"]
            for cat_name, group in grouped:
                cat_desc = group.describe().to_dict()
                by_category[str(cat_name)] = {k: round(float(v), 4) for k, v in cat_desc.items()}

        # Build box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if category_col:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Text Length Distribution by Category")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "by_category": by_category
        }

        session.store_result("describe_data_tool", {"text_column": text_column}, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        describe_data_tool
    ]
