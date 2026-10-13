import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """
    Test build_dtm_tool against Known Answer 4 in TEST_FIXTURE.md.
    
    1. Loads agent_dev/sample_fixture.csv using premade load_dataset_tool.
    2. Runs build_dtm_tool with default arguments.
    3. Verifies:
       - Vocabulary terms: ['alpha', 'always', 'beta', 'delta', 'gamma'] (5 terms total).
       - Matrix shape: 8 rows x 5 columns.
       - Non-zero count: 21.
       - Total elements: 40.
       - Sparsity percentage: 47.5%.
    """
    session = SessionState()

    # Load premade load_dataset_tool and run it on sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's build_dtm_tool directly from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool with default parameters
    result = build_dtm_tool.invoke({})

    # Verify return dictionary values against Known Answer 4
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Verify session state feature names and matrix shape
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_matrix.nnz == 21
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    """
    Test term_frequency_tool against Known Answer 14 in TEST_FIXTURE.md.

    1. Loads agent_dev/sample_fixture.csv using premade load_dataset_tool.
    2. Builds global DTM with build_dtm_tool.
    3. Runs term_frequency_tool with top_n=10.
    4. Verifies frequencies against Known Answer 14:
       - alpha: 6
       - always: 7
       - beta: 3
       - delta: 4
       - gamma: 7
    5. Verifies total word count = 27 (6+7+3+4+7).
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get term_frequency_tool
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    result = term_frequency_tool.invoke({"top_n": 10})

    # Assertions against Known Answer 14
    assert result["total_terms"] == 5
    assert result["total_word_count"] == 27

    expected_top_terms = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    assert result["top_terms"] == expected_top_terms

    # Check full report in session
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    assert list(full_report.columns[:2]) == ["term", "frequency"]
    report_freqs = dict(zip(full_report["term"], full_report["frequency"]))
    assert report_freqs == expected_top_terms


def test_dtm_heatmap_tool_known_answer_21():
    """
    Test dtm_heatmap_tool against Known Answer 21 in TEST_FIXTURE.md.

    1. Loads agent_dev/sample_fixture.csv using premade load_dataset_tool.
    2. Builds global DTM with build_dtm_tool.
    3. Runs dtm_heatmap_tool with default bounds (n_terms=20, n_documents=20).
       - Verifies full 8x5 matrix values and terms.
       - Verifies figure is attached to session.pending_figure.
    4. Runs dtm_heatmap_tool with sliced bounds (n_terms=2, n_documents=3).
       - Verifies sliced matrix is [[3, 1], [2, 1], [1, 1]].
    """
    session = SessionState()

    # Load dataset & build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")
    
    build_dtm_tool.invoke({})

    # 1. Test full matrix slice (defaults)
    full_res = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})
    assert full_res["n_documents"] == 8
    assert full_res["n_terms"] == 5
    assert full_res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert full_res["matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # 2. Test corner slice (3x2 corner)
    slice_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_res["n_documents"] == 3
    assert slice_res["n_terms"] == 2
    assert slice_res["terms"] == ["alpha", "always"]
    assert slice_res["matrix"] == [[3, 1], [2, 1], [1, 1]]
