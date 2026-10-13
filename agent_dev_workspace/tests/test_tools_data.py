import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_list_files_tool_newdataset():
    """
    Tests list_files_tool against real repo state in 'newdataset' subdirectory (Known answer #11).
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert "Reddit-stock-sentiment.csv" in res["files"]
    assert "result_id" in res


def test_inspect_data_tool():
    """
    Tests inspect_data_tool on sample_fixture.csv (Known answer #12).
    inspect_data_tool deliberately has no fixed contract in specs, so we test
    that it correctly peeks at sample_fixture.csv loaded into session.dataframe.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 3})

    assert res["n_rows"] == 3
    assert res["total_rows"] == 8
    assert "text" in res["columns"]
    assert len(res["sample_records"]) == 3
    assert res["sample_records"][0]["text"] == "always alpha alpha alpha beta"


def test_check_missing_tool():
    """
    Tests check_missing_tool on sample_fixture.csv (Known answer #2).
    Exactly 1 row has missing/empty text (row 7).
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"column": "text"})

    assert res["missing_count"] == 1
    assert res["total_rows"] == 8
    assert res["missing_indices"] == [7]


def test_check_duplicates_tool():
    """
    Tests check_duplicates_tool on sample_fixture.csv (Known answer #3).
    Rows 5 and 6 are duplicate ("always gamma delta").
    Check drop=False leaves 8 rows, drop=True leaves 6 rows (drops both copies).
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check without dropping
    res_check = check_duplicates_tool.invoke({"subset": "text", "drop": False})
    assert res_check["duplicate_count"] == 2
    assert res_check["total_rows"] == 8
    assert res_check["remaining_rows"] == 8
    assert res_check["dropped"] is False

    # Check with dropping (keep=False removes both copies)
    res_drop = check_duplicates_tool.invoke({"subset": "text", "drop": True})
    assert res_drop["duplicate_count"] == 2
    assert res_drop["total_rows"] == 8
    assert res_drop["remaining_rows"] == 6
    assert res_drop["dropped"] is True


def test_tokenize_tool():
    """
    Tests tokenize_tool on sample_fixture.csv (Known answer #10).
    Verifies that unigrams are created per document and stored in session state.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"column": "text"})

    # Known answer #10 verification
    assert res["total_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["vocabulary_size"] == 5

    # Check document 0 tokens
    doc0_tokens = session.dataframe["unigrams"].iloc[0]
    assert doc0_tokens == ["always", "alpha", "alpha", "alpha", "beta"]

    # Check row 7 (empty text) returns an empty token list
    doc7_tokens = session.dataframe["unigrams"].iloc[7]
    assert doc7_tokens == []

    # Check artifacts update
    assert "unigrams" in session.artifacts
    assert len(session.artifacts["unigrams"]) == 8


def test_sample_data_tool():
    """
    Tests sample_data_tool on sample_fixture.csv (Known answer #13).
    Calling with n=4, random_state=42 returns exactly rows [1, 5, 0, 7].
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["original_rows"] == 8
    assert res["sampled_rows"] == 4
    assert res["sample_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_tool():
    """
    Tests describe_data_tool on sample_fixture.csv against Known answer #18.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text"})

    # Known Answer #18 expected values
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert overall["max"] == 30

    by_cat = res["by_category"]
    assert "catA" in by_cat
    assert "catB" in by_cat

    # catA stats
    catA = by_cat["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert catA["min"] == 0
    assert np.isclose(catA["25%"], 12.75, atol=1e-3)
    assert np.isclose(catA["50%"], 20.0, atol=1e-3)
    assert np.isclose(catA["75%"], 24.5, atol=1e-3)
    assert catA["max"] == 29

    # catB stats
    catB = by_cat["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["min"] == 18
    assert np.isclose(catB["25%"], 18.0, atol=1e-3)
    assert np.isclose(catB["50%"], 21.0, atol=1e-3)
    assert np.isclose(catB["75%"], 25.5, atol=1e-3)
    assert catB["max"] == 30

    # Confirm box plot figure was stored in session.pending_figure
    assert session.pending_figure is not None
