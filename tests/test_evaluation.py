
from travel_agent.evaluation import (
    EVALUATION_SCENARIOS,
    check_expected_values,
    normalize_response,
)

from travel_agent.evaluation import evaluate_scenario


def test_normalize_response_removes_thousands_separators():
    assert normalize_response("EGP 7,500") == "egp 7500"


def test_check_expected_values_accepts_matching_response():
    assert check_expected_values(
        "The total is EGP 7,500.",
        ["7500"],
    )


def test_check_expected_values_rejects_missing_value():
    assert not check_expected_values(
        "The total is EGP 7,500.",
        ["7500", "4800"],
    )


def test_six_original_evaluation_scenarios_exist():
    assert len(EVALUATION_SCENARIOS) == 6



def test_evaluator_detects_missing_required_phrase():
    scenario = {
        "expected_values": [],
        "required_phrases": ["alexandria"],
        "forbidden_phrases": [],
    }

    result = evaluate_scenario(
        "The Red Sea is a popular winter destination.",
        scenario,
    )

    assert result["passed"] is False


def test_evaluator_detects_forbidden_phrase():
    scenario = {
        "expected_values": [],
        "required_phrases": [],
        "forbidden_phrases": ["guaranteed free hotels"],
    }

    result = evaluate_scenario(
        "You can find guaranteed free hotels everywhere.",
        scenario,
    )

    assert result["passed"] is False
    assert result["forbidden_phrases_found"] == [
        "guaranteed free hotels"
    ]
