import pytest
from quality_metrics import (
    extract_thinking_content,
    calculate_security_score,
    has_critical_security_failures,
    get_detailed_security_report,
)

SAMPLE_RESPONSE = """
<think>
1. Validate inputs: sanitize and validate user inputs
2. Use encryption to protect sensitive data
</think>
"""

def test_extract_thinking_content():
    thinking = extract_thinking_content(SAMPLE_RESPONSE)
    assert "Validate inputs" in thinking


def test_calculate_security_score_and_report():
    thinking = extract_thinking_content(SAMPLE_RESPONSE)
    score = calculate_security_score(thinking)
    assert 0.0 <= score <= 1.0

    report = get_detailed_security_report(thinking)
    assert "overall_score" in report
    # Since the thinking mentions validation & encryption, we expect no critical failures
    assert report["overall_score"] > 0


def test_has_critical_security_failures_false():
    thinking = extract_thinking_content(SAMPLE_RESPONSE)
    assert not has_critical_security_failures(thinking)
