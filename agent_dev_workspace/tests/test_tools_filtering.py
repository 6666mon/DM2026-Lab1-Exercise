import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """
    Test variance_filter_tool against Known Answer 5 in TEST_FIXTURE.md.

    1. Loads agent_dev/sample_fixture.csv with load_dataset_tool.
    2. Builds global DTM with build_dtm_tool (default settings).
    3. Verifies exact variances per term:
       - alpha: 1.1875
       - always: 0.109375
       - beta: 0.234375
       - delta: 0.25
       - gamma: 1.109375
    4. Applies variance_filter_tool with threshold=0.15:
       - Kept terms: ['alpha', 'beta', 'delta', 'gamma']
       - Removed terms: ['always']
    """
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold 0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assertions against Known Answer 5
    assert result["threshold"] == 0.15
    assert result["total_terms"] == 5
    assert result["retained_terms_count"] == 4
    assert result["removed_terms_count"] == 1
    assert set(result["retained_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms"] == ["always"]

    # Verify report stored in session for visualize_result_tool
    last_res_id = result["result_id"]
    report_df = session.results_store[last_res_id]["full_report"]
    
    # Check column order contract: Column 1 = term, Column 2 = variance
    assert list(report_df.columns[:2]) == ["term", "variance"]
    
    # Check reported numeric variances against Known Answer 5
    variances = dict(zip(report_df["term"], report_df["variance"]))
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-4)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-4)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-4)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-4)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-4)


def test_pearson_filter_tool_known_answer_6():
    """
    Test pearson_filter_tool against Known Answer 6 in TEST_FIXTURE.md.

    1. Loads agent_dev/sample_fixture.csv with load_dataset_tool.
    2. Builds global DTM with build_dtm_tool.
    3. Runs pearson_filter_tool with target_class="catB".
    4. Asserts exact Pearson r correlations per term against Known Answer 6:
       - alpha: -0.6882
       - always: 0.3780
       - beta: -0.7746
       - delta: 1.0000
       - gamma: 0.8307
    """
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool with target_class="catB" and threshold=0.0
    result = pearson_filter_tool.invoke({
        "target_class": "catB",
        "threshold": 0.0
    })

    # Check full_report format and values against Known Answer 6
    last_res_id = result["result_id"]
    report_df = session.results_store[last_res_id]["full_report"]

    assert list(report_df.columns[:2]) == ["term", "pearson_r"]

    pearson_map = dict(zip(report_df["term"], report_df["pearson_r"]))
    assert pearson_map["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert pearson_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert pearson_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert pearson_map["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert pearson_map["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Check filtering behavior with threshold=0.0 (keeps terms with r >= 0.0)
    assert set(result["retained_terms"]) == {"always", "delta", "gamma"}
    assert set(result["removed_terms"]) == {"alpha", "beta"}


def test_spearman_filter_tool_known_answer_6():
    """
    Test spearman_filter_tool against Known Answer 6 in TEST_FIXTURE.md.

    1. Loads agent_dev/sample_fixture.csv with load_dataset_tool.
    2. Builds global DTM with build_dtm_tool.
    3. Runs spearman_filter_tool with target_class="catB".
    4. Asserts exact Spearman rho correlations per term against Known Answer 6:
       - alpha: -0.7500
       - always: 0.3780
       - beta: -0.7746
       - delta: 1.0000
       - gamma: 0.9363
    """
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool with target_class="catB" and threshold=0.0
    result = spearman_filter_tool.invoke({
        "target_class": "catB",
        "threshold": 0.0,
        "abs_value": False
    })

    # Check full_report format and values against Known Answer 6
    last_res_id = result["result_id"]
    report_df = session.results_store[last_res_id]["full_report"]

    assert list(report_df.columns[:2]) == ["term", "spearman_r"]

    spearman_map = dict(zip(report_df["term"], report_df["spearman_r"]))
    assert spearman_map["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert spearman_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert spearman_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert spearman_map["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert spearman_map["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Check filtering behavior with threshold=0.0
    assert set(result["retained_terms"]) == {"always", "delta", "gamma"}
    assert set(result["removed_terms"]) == {"alpha", "beta"}
