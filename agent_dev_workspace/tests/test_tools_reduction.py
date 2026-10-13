"""Tests for tools_reduction.py using sample_fixture.csv."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def _setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using CountVectorizer to populate session.feature_matrix
    vec = CountVectorizer()
    X = vec.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_matrix = X
    session.feature_names = vec.get_feature_names_out().tolist()

    if "category_name" in session.dataframe.columns:
        session.labels = session.dataframe["category_name"].values
    elif "label" in session.dataframe.columns:
        session.labels = session.dataframe["label"].values
        session.dataframe["category_name"] = session.dataframe["label"]

    tools = make_tools(session)
    tool_dict = {t.name: t for t in tools}
    return session, tool_dict


def test_reduce_dimensions_pca():
    """Test reduce_dimensions_tool PCA against known answer #7."""
    session, tools = _setup_session_with_dtm()
    reduce_tool = tools["reduce_dimensions_tool"]

    result = reduce_tool.invoke({"method": "pca", "n_components": 2})

    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_variance = [0.7532, 0.1965]

    coords = np.array(result["coordinates"])
    variance = result["explained_variance_ratio"]

    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)
    np.testing.assert_allclose(variance, expected_variance, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    """Test reduce_dimensions_tool t-SNE."""
    session, tools = _setup_session_with_dtm()
    reduce_tool = tools["reduce_dimensions_tool"]

    result = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })

    assert "coordinates" in result
    coords = np.array(result["coordinates"])

    assert coords.shape == (8, 2)
    assert np.all(np.isfinite(coords))
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    """Test reduce_dimensions_tool UMAP against known answer #9."""
    session, tools = _setup_session_with_dtm()
    reduce_tool = tools["reduce_dimensions_tool"]

    result = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])

    coords = np.array(result["coordinates"])
    np.testing.assert_allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None


def test_binarize_labels():
    """Test binarize_labels_tool on sample_fixture categories."""
    session, tools = _setup_session_with_dtm()
    binarize_tool = tools["binarize_labels_tool"]

    result = binarize_tool.invoke({})

    assert result["classes"] == ["catA", "catB"]
    assert result["shape"] == [8, 2]

    expected_matrix = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
    ])

    binary_matrix = np.array(result["binarized_labels"])
    np.testing.assert_array_equal(binary_matrix, expected_matrix)
    assert np.array_equal(session.artifacts["binarized_labels"], expected_matrix)