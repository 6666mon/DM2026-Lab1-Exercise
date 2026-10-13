import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_fpgrowth_known_answer_17():
    """
    Tests mine_patterns_tool with algorithm="fpgrowth" on sample_fixture.csv (Known answer #17).
    - filtering_method="variance" and "term_frequency" keep zero terms for both categories.
    - filtering_method="tfidf" keeps catA -> 'alpha' (support 3 at min_sup=1..3) and catB -> 'gamma' (support 4 at min_sup=1..4).
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
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Test zero-terms surviving for variance filtering
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["pattern_count"] == 0
    assert res_var["patterns"] == []

    # Test tfidf filtering for catA
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["pattern_count"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # Test tfidf filtering for catB
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB["pattern_count"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    """
    Tests mine_patterns_tool with algorithm="topk" and "maxfpgrowth" on sample_fixture.csv (Known answer #20).
    - topk algorithm (k=1, 2, or 3) returns 1 pattern per category: catA -> 'alpha' (sup 3), catB -> 'gamma' (sup 4).
    - maxfpgrowth returns 0 patterns for both categories at min_sup=1 or 4.
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
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Test topk algorithm on catA
    res_topk_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA["pattern_count"] == 1
    assert res_topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_catA["patterns"][0]["support"] == 3

    # Test topk algorithm on catB
    res_topk_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 2
    })
    assert res_topk_catB["pattern_count"] == 1
    assert res_topk_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_topk_catB["patterns"][0]["support"] == 4

    # Test maxfpgrowth algorithm on catA (returns 0 patterns)
    res_max_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catA["pattern_count"] == 0
    assert res_max_catA["patterns"] == []

    # Test maxfpgrowth algorithm on catB (returns 0 patterns)
    res_max_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catB["pattern_count"] == 0
    assert res_max_catB["patterns"] == []
