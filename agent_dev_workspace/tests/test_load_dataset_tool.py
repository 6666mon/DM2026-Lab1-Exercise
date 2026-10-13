import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools

def test_load_dataset_tool_known_answer_1():
    """
    Test load_dataset_tool against Known Answer 1 in TEST_FIXTURE.md.
    Loading agent_dev/sample_fixture.csv should result in:
    - 8 total documents (n_documents == 8)
    - Category split: 4 catA, 4 catB (counts_per_category == {"catA": 4, "catB": 4})
    """
    # Create a fresh session state instance
    session = SessionState()
    
    # Load premade tools bound to this session
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Execute load_dataset_tool with the test fixture parameters
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Verify the return dictionary fields matching Known Answer 1
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
