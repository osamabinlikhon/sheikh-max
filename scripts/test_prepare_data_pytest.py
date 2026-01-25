import pytest
from prepare_data import create_sample_dataset


def test_create_sample_dataset_structure():
    ds = create_sample_dataset()
    # Expect at least 3 samples as defined in the script
    assert len(ds) >= 3

    sample = ds[0]
    assert "messages" in sample
    # Ensure assistant message includes thinking and content fields
    assistant_msgs = [m for m in sample["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs
    first_assistant = assistant_msgs[0]
    assert "thinking" in first_assistant
    assert "content" in first_assistant
