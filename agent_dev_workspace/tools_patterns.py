import os
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Import PAMI pattern mining tools
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    """
    Factory function that binds session state and returns pattern mining tools.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns for a specific category using PAMI algorithms.

        Filters session.dataframe to a single category, constructs a category-local DTM,
        applies term filtering (variance, tfidf, or term_frequency), converts data to a
        transactional database, and executes FPGrowth, TopK (FAE), or MaxFPGrowth mining.

        Args:
            category_name: Target category label to mine patterns for (e.g. "catA").
            filtering_method: Term filtering method ("variance", "tfidf", or "term_frequency"). Default is "tfidf".
            algorithm: PAMI pattern mining algorithm ("fpgrowth", "topk", or "maxfpgrowth"). Default is "fpgrowth".
            min_sup: Minimum support count threshold for "fpgrowth" or "maxfpgrowth".
            k: Top-k patterns count requirement for "topk" algorithm.

        Returns:
            A summary dictionary containing result_id, category_name, filtering_method,
            algorithm, patterns list, and pattern_count.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        # Determine category column name
        cat_col = "category_name" if "category_name" in session.dataframe.columns else ("category" if "category" in session.dataframe.columns else None)
        if cat_col is None:
            raise ValueError("No category column found in session.dataframe.")

        # Filter dataframe to requested category
        cat_df = session.dataframe[session.dataframe[cat_col] == category_name]
        if len(cat_df) == 0:
            raise ValueError(f"Category '{category_name}' not found in session.dataframe[{cat_col}].")

        # Step 1: Build fresh category-local DTM with English stop words
        texts = cat_df["text"].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out().tolist()
        except ValueError:
            # Vectorizer raises ValueError if all words are stop words or vocabulary is empty
            dtm = None
            feature_names = []

        if dtm is None or len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dense_dtm = dtm.toarray()
        df_dtm = pd.DataFrame(dense_dtm, columns=feature_names)

        # Step 2: Apply percentile filtering method
        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                scores = df_dtm.var(axis=0, ddof=0).values
            else:
                scores = df_dtm.sum(axis=0).values

            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            # Keep terms strictly between 5th and 95th percentiles (middle 90%)
            keep_mask = (scores > p5) & (scores < p95)
            kept_cols = [col for col, keep in zip(feature_names, keep_mask) if keep]

        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dense_dtm).toarray()
            scores = tfidf_matrix.mean(axis=0)
            p20 = np.percentile(scores, 20)
            # Keep terms at or above the 20th percentile (top 80%)
            keep_mask = scores >= p20
            kept_cols = [col for col, keep in zip(feature_names, keep_mask) if keep]
        else:
            raise ValueError(f"Unknown filtering_method '{filtering_method}'. Expected 'variance', 'tfidf', or 'term_frequency'.")

        # Step 3: Handle zero terms surviving filtering
        if len(kept_cols) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        df_filtered = df_dtm[kept_cols]

        # Filter out empty documents with no surviving terms
        non_empty_mask = (df_filtered.sum(axis=1) > 0)
        df_filtered = df_filtered[non_empty_mask]

        if len(df_filtered) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no documents contain surviving terms"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert to transactional database file and run PAMI algorithm
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "transactional.db")

            # Convert dataframe to transactional database via PAMI DF2DB
            converter = DF2DB.DF2DB(df_filtered, "dense")
            converter.convert2TransactionalDatabase(db_path, ">=", 1)

            if algorithm == "fpgrowth":
                if min_sup is None:
                    raise ValueError("min_sup is required for algorithm 'fpgrowth'.")
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                if k is None:
                    raise ValueError("k is required for algorithm 'topk'.")
                miner = FAE.FAE(db_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    raise ValueError("min_sup is required for algorithm 'maxfpgrowth'.")
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")
            else:
                raise ValueError(f"Unknown algorithm '{algorithm}'. Expected 'fpgrowth', 'topk', or 'maxfpgrowth'.")

            miner.startMine()
            pami_df = miner.getPatternsAsDataFrame()

        patterns_list = []
        if pami_df is not None and len(pami_df) > 0:
            for _, row in pami_df.iterrows():
                p_str = str(row["Patterns"]).strip()
                sup_val = int(row["Support"])
                pattern_items = [item.strip() for item in p_str.split("\t") if item.strip()]
                patterns_list.append({
                    "pattern": pattern_items,
                    "support": sup_val
                })

        # Step 5: Store result and artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns_list[:50],
            "pattern_count": len(patterns_list)
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary
        )

        return summary

    return [mine_patterns_tool]
