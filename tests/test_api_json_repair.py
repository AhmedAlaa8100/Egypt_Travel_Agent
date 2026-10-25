import json
from datetime import date
from unittest.mock import Mock

from fastapi.testclient import TestClient

from travel_agent.api import (
    TripPlanRequest,
    _apply_baseline_estimate,
    _build_structured_response,
    _extract_json,
)
from travel_agent.api import app as api_app


def _request(**overrides):
    values = {
        "destination": "Alexandria, Egypt",
        "start_date": date(2026, 11, 12),
        "duration_days": 4,
        "travelers": 2,
        "travel_style": "Culture & history",
        "budget_amount": 18000,
        "budget_currency": "EGP",
        "thread_id": "repair-thread",
    }
    values.update(overrides)
    return TripPlanRequest(**values)


MALFORMED_PLAN = """Here is your plan:
```json
{
  "trip_summary": {"destination": "Alexandria, Egypt", "start_date": "2026-11-12"},
  "budget": {
    "total_egp": 18000,
    "breakdown": [
      {"category": "Accommodation", "amount_egp": "8,400", "note": "3 nights", "share_percent": 46.67},
      {"category": "Meals", "amount_egp": 2000,}
    ]
  },
  "itinerary": [
    {"day": 1, "title": "Arrival", "summary": "Settle in.", "activities": ["Walk"], "duration": "Evening"},
    {"day": 2", "title": "Museums", "summary": "Explore.", "activities": ["Library"], "duration": "Full day"}
  ],
  "warnings": ["Estimates only."],
  "answer": "A four-day plan with a readable summary."
}
```
Enjoy your trip.
"""


def test_extract_json_repairs_stray_quote_and_trailing_comma():
    data = _extract_json(MALFORMED_PLAN)

    assert data is not None
    assert data["trip_summary"]["destination"] == "Alexandria, Egypt"
    assert data["budget"]["total_egp"] == 18000
    assert data["itinerary"][1]["day"] == 2
    assert data["answer"] == "A four-day plan with a readable summary."


def test_extract_json_prefers_a_plan_object_over_an_earlier_note():
    text = (
        '{"note": "thinking out loud"}\n'
        '{"trip_summary": {"destination": "Aswan"}, "budget": {"total_egp": 100}}'
    )

    data = _extract_json(text)

    assert data["trip_summary"]["destination"] == "Aswan"


def test_extract_json_handles_fenced_json_surrounded_by_prose():
    text = 'Here is the plan:\n```json\n{"a": 1}\n```\nHope that helps.'

    assert _extract_json(text) == {"a": 1}


def test_extract_json_handles_smart_quotes_and_trailing_comma():
    assert _extract_json('{\u201ca\u201d: 1,}') == {"a": 1}


def test_extract_json_closes_a_truncated_object():
    data = _extract_json('{"trip_summary": {"destination": "Luxor"')

    assert data == {"trip_summary": {"destination": "Luxor"}}


def test_structured_response_recovers_malformed_agent_plan():
    response = _build_structured_response(_request(), MALFORMED_PLAN)

    assert response.budget.total_egp == 18000
    assert response.budget.breakdown_total_egp == 10400
    assert len(response.budget.breakdown) == 2
    assert response.budget.breakdown[0].amount_egp == 8400
    assert response.budget.breakdown[0].share_percent == 46
    assert response.budget.breakdown[1].share_percent == 19
    assert len(response.itinerary) == 2
    assert response.itinerary[1].title == "Museums"
    assert response.answer == "A four-day plan with a readable summary."
    assert '"trip_summary"' not in response.answer
    assert response.warnings == ["Estimates only."]


def test_structured_response_computes_missing_shares():
    answer = json.dumps(
        {
            "budget": {
                "breakdown": [
                    {"category": "Stay", "amount_egp": 6000},
                    {"category": "Food", "amount_egp": 2000},
                    {"category": "Transport", "amount_egp": 2000},
                ]
            },
            "answer": "Ready.",
        }
    )

    response = _build_structured_response(_request(budget_amount=12000), answer)

    assert response.budget.total_egp == 10000
    assert response.budget.breakdown_total_egp == 10000
    assert [entry.share_percent for entry in response.budget.breakdown] == [
        60,
        20,
        20,
    ]


def test_structured_response_survives_wrong_field_types():
    answer = json.dumps(
        {
            "trip_summary": "not a dict",
            "budget": "not a dict",
            "itinerary": "not a list",
            "warnings": 123,
            "answer": "A safe answer.",
        }
    )

    response = _build_structured_response(_request(), answer)

    assert response.answer == "A safe answer."
    assert response.budget.total_egp == 0
    assert response.budget.breakdown == []
    assert response.itinerary == []
    assert len(response.warnings) == 1
    assert "numeric total cost" in response.warnings[0]


def test_unreadable_json_does_not_leak_into_the_answer():
    answer = '{"trip_summary": {oops}, "answer": "Could not finish the plan."}'

    response = _build_structured_response(_request(), answer)

    assert response.answer == "Could not finish the plan."
    assert '"trip_summary"' not in response.answer
    assert len(response.itinerary) == 1
    assert '"trip_summary"' not in response.itinerary[0].summary
    assert any("unstructured" in warning.lower() for warning in response.warnings)


def test_structured_response_normalises_order_and_string_fields():
    answer = json.dumps(
        {
            "budget": {
                "breakdown": {
                    "category": "Stay",
                    "amount_egp": 5000,
                }
            },
            "itinerary": [
                {"day": 5, "title": "Second", "activities": "Walk; Eat"},
                {"day": 2, "title": "First", "activities": ["Swim"]},
            ],
            "warnings": "One warning",
            "answer": "Ready.",
        }
    )

    response = _build_structured_response(_request(), answer)

    assert [entry.day for entry in response.itinerary] == [1, 2]
    assert [entry.title for entry in response.itinerary] == ["First", "Second"]
    assert response.itinerary[0].activities == ["Swim"]
    assert response.itinerary[1].activities == ["Walk", "Eat"]
    assert response.budget.total_egp == 5000
    assert response.budget.breakdown_total_egp == 5000
    assert len(response.budget.breakdown) == 1
    assert response.warnings == ["One warning"]


def test_endpoint_returns_structured_data_for_malformed_json():
    import travel_agent.api as api_module

    agent = Mock()
    agent.run.return_value = MALFORMED_PLAN
    original = api_module.get_agent
    api_module.get_agent = lambda: agent
    try:
        client = TestClient(api_app)
        resp = client.post(
            "/api/plan",
            json={
                "destination": "Alexandria, Egypt",
                "start_date": "2026-11-12",
                "duration_days": 4,
                "travelers": 2,
                "travel_style": "Culture & history",
                "budget_amount": 18000,
                "budget_currency": "EGP",
                "thread_id": "endpoint-repair-thread",
            },
        )
    finally:
        api_module.get_agent = original

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["budget"]["total_egp"] == 18000
    assert len(data["itinerary"]) == 2
    assert data["answer"] == "A four-day plan with a readable summary."


def test_structured_response_drops_budget_placeholder_row():
    answer = json.dumps(
        {
            "budget": {
                "total_egp": 50000,
                "breakdown": [
                    {
                        "category": "Accommodation",
                        "amount_egp": 11200,
                        "note": "4 nights",
                    },
                    {
                        "category": "Other",
                        "amount_egp": 38800,
                        "note": "Placeholder for flights, meals and transport, which are not included.",
                    },
                ],
            },
            "answer": "Plan ready.",
        }
    )

    response = _build_structured_response(_request(budget_amount=50000), answer)

    assert response.budget.total_egp == 11200
    assert response.budget.breakdown_total_egp == 11200
    assert [entry.category for entry in response.budget.breakdown] == [
        "Accommodation"
    ]
    assert any("placeholder" in warning.lower() for warning in response.warnings)


def test_structured_response_uses_itemised_total_when_reported_total_is_lower():
    answer = json.dumps(
        {
            "budget": {
                "total_egp": 1000,
                "breakdown": [
                    {"category": "Stay", "amount_egp": 6000},
                    {"category": "Food", "amount_egp": 2000},
                ],
            },
            "answer": "Plan ready.",
        }
    )

    response = _build_structured_response(_request(budget_amount=20000), answer)

    assert response.budget.total_egp == 8000
    assert response.budget.breakdown_total_egp == 8000
    assert any("lower" in warning.lower() for warning in response.warnings)


def test_clean_text_normalises_non_breaking_typography():
    from travel_agent.api import _clean_text

    cleaned = _clean_text("A 4\u2011day trip with\u202fEGP\u00a010,000.\u200b")

    assert cleaned == "A 4-day trip with EGP 10,000."


def test_endpoint_retries_an_empty_agent_response():
    import travel_agent.api as api_module
    valid_plan = json.dumps(
        {
            "budget": {
                "total_egp": 14650,
                "breakdown": [
                    {"category": "Stay", "amount_egp": 8650},
                    {"category": "Food", "amount_egp": 6000},
                ],
            },
            "itinerary": [{"day": 1, "title": "Arrive", "summary": "Settle in."}],
            "answer": "Your plan is ready.",
        }
    )
    agent = Mock()
    agent.run.side_effect = ["", valid_plan]
    original = api_module.get_agent
    api_module.get_agent = lambda: agent
    try:
        client = TestClient(api_app)
        resp = client.post(
            "/api/plan",
            json={
                "destination": "Alexandria, Egypt",
                "start_date": "2026-11-12",
                "duration_days": 4,
                "travelers": 2,
                "travel_style": "Culture & history",
                "budget_amount": 18000,
                "budget_currency": "EGP",
                "thread_id": "retry-thread",
            },
        )
    finally:
        api_module.get_agent = original

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["budget"]["total_egp"] == 14650
    assert data["thread_id"] == "retry-thread"
    assert agent.run.call_count == 2
    first_call, second_call = agent.run.call_args_list
    assert first_call.kwargs["thread_id"] == "retry-thread"
    assert second_call.kwargs["thread_id"] != "retry-thread"


def test_baseline_estimate_normalises_destination_and_explains_limits():
    request = _request(
        destination="Alexandria, Egypt",
        duration_days=4,
        travelers=2,
        budget_amount=18000,
    )
    response = _build_structured_response(request, "plain text")

    result = _apply_baseline_estimate(request, response)

    # 3 nights at the Alexandria prototype rate, one shared room.
    assert result.budget.total_egp == 8400
    assert result.budget.breakdown_total_egp == 8400
    assert result.budget.within_budget is True
    assert [entry.category for entry in result.budget.breakdown] == [
        "Accommodation"
    ]
    assert any("baseline accommodation" in warning.lower() for warning in result.warnings)


def test_baseline_estimate_is_skipped_for_unsupported_destinations():
    request = _request(destination="Paris, France")
    response = _build_structured_response(request, "plain text")

    result = _apply_baseline_estimate(request, response)

    assert result.budget.total_egp == 0
    assert result.budget.breakdown == []
