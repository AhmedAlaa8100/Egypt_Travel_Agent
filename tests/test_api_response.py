import json as pyjson
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from travel_agent.api import (
    BudgetCard,
    CostEntry,
    ItineraryDay,
    TripPlanResponse,
    TripSummary,
    build_agent_prompt,
    _build_structured_response,
    _extract_json,
)
from travel_agent.api import app as api_app
from travel_agent.api import TripPlanRequest
from datetime import date


def _client_with_agent(agent):
    import travel_agent.api as api_module

    api_module.get_agent = lambda: agent
    return TestClient(api_app)


def test_trip_plan_response_has_structured_fields():
    response = TripPlanResponse(
        trip_summary=TripSummary(
            destination="Alexandria, Egypt",
            start_date="2026-11-12",
            duration_days=4,
            travelers=2,
            travel_style="Culture & history",
            budget_amount_egp=18000,
        ),
        budget=BudgetCard(
            total_egp=14650,
            currency="EGP",
            budget_amount_egp=18000,
            within_budget=True,
            utilization_percent=81.4,
            breakdown=[
                CostEntry(category="Stay", amount_egp=5100, note="3 nights", share_percent=35),
                CostEntry(category="Food", amount_egp=3200, note="Local cafés", share_percent=22),
            ],
        ),
        itinerary=[
            ItineraryDay(
                day=1,
                title="Arrive",
                summary="Settle into the city.",
                activities=["waterfront walk"],
                duration="3-4 hrs",
            )
        ],
        warnings=["Estimates are not live quotes."],
        answer="Your Alexandria plan is ready.",
        thread_id="test-thread",
    )
    assert response.trip_summary.destination == "Alexandria, Egypt"
    assert response.budget.within_budget is True
    assert response.budget.utilization_percent == 81.4
    assert len(response.itinerary) == 1
    assert response.itinerary[0].day == 1
    assert response.warnings == ["Estimates are not live quotes."]
    assert response.answer == "Your Alexandria plan is ready."
    assert response.thread_id == "test-thread"


def test_build_agent_prompt_requests_structured_json():
    request = TripPlanRequest(
        destination="Alexandria, Egypt",
        start_date=date(2026, 11, 12),
        duration_days=4,
        travelers=2,
        travel_style="Culture & history",
        budget_amount=18000,
        budget_currency="EGP",
        thread_id="test-thread-1",
    )
    prompt = build_agent_prompt(request)

    assert "Alexandria, Egypt" in prompt
    assert "2026-11-12" in prompt
    assert "Duration: 4 days" in prompt
    assert "Travelers: 2" in prompt
    assert "18,000 EGP" in prompt
    assert "do not present a partial cost as the full trip total" in prompt
    assert "trip_summary" in prompt
    assert "budget" in prompt
    assert "itinerary" in prompt
    assert "warnings" in prompt
    assert "answer" in prompt


def test_extract_json_handles_code_fences_and_plain_text():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json("Just plain text, no json here.") is None
    assert _extract_json("") is None


def test_structured_response_fills_defaults_for_missing_fields():
    request = TripPlanRequest(
        destination="Luxor",
        start_date=None,
        duration_days=3,
        travelers=1,
        travel_style="Relaxed getaway",
        budget_amount=10000,
        budget_currency="EGP",
        thread_id="t",
    )
    response = _build_structured_response(request, "some unstructured text")

    assert response.trip_summary.destination == "Luxor"
    assert response.trip_summary.start_date is None
    assert response.trip_summary.duration_days == 3
    assert response.trip_summary.travelers == 1
    assert response.trip_summary.travel_style == "Relaxed getaway"
    assert response.trip_summary.budget_amount_egp == 10000
    assert response.budget.total_egp == 0
    assert response.budget.budget_amount_egp == 10000
    assert response.budget.within_budget is True
    assert response.budget.utilization_percent == 0.0
    assert len(response.itinerary) == 1
    assert response.itinerary[0].day == 1
    assert response.warnings == [
        "The agent returned an unstructured response; no reliable cost estimate is available."
    ]
    assert response.answer == "some unstructured text"
    assert response.thread_id == "t"


def test_structured_response_computes_budget_server_side():
    request = TripPlanRequest(
        destination="Aswan",
        start_date=date(2026, 11, 12),
        duration_days=3,
        travelers=2,
        travel_style="Culture & history",
        budget_amount=5000,
        budget_currency="EGP",
        thread_id="t",
    )
    # The agent claims a huge total and says within_budget=True; the server
    # must still compute utilization and within_budget from real figures.
    answer = pyjson.dumps(
        {
            "trip_summary": {
                "destination": "Aswan",
                "duration_days": 3,
                "travelers": 2,
                "travel_style": "Culture & history",
                "budget_amount_egp": 5000,
            },
            "budget": {
                "total_egp": 12000,
                "currency": "EGP",
                "within_budget": True,
            },
            "itinerary": [],
            "warnings": [],
            "answer": "Plan ready.",
        }
    )
    response = _build_structured_response(request, answer)

    assert response.budget.total_egp == 12000
    assert response.budget.budget_amount_egp == 5000
    assert response.budget.within_budget is False
    assert response.budget.utilization_percent == 240.0


def test_structured_response_parses_agent_json():
    request = TripPlanRequest(
        destination="Alexandria, Egypt",
        start_date=date(2026, 11, 12),
        duration_days=4,
        travelers=2,
        travel_style="Culture & history",
        budget_amount=18000,
        budget_currency="EGP",
        thread_id="t",
    )
    answer = pyjson.dumps(
        {
            "trip_summary": {
                "destination": "Alexandria, Egypt",
                "start_date": "2026-11-12",
                "duration_days": 4,
                "travelers": 2,
                "travel_style": "Culture & history",
                "budget_amount_egp": 18000,
            },
            "budget": {
                "total_egp": 14650,
                "currency": "EGP",
                "budget_amount_egp": 18000,
                "within_budget": True,
                "breakdown": [
                    {"category": "Stay", "amount_egp": 5100, "note": "3 nights", "share_percent": 35},
                    {"category": "Food", "amount_egp": 3200, "note": "Local cafés", "share_percent": 22},
                ],
            },
            "itinerary": [
                {
                    "day": 1,
                    "title": "Arrive",
                    "summary": "Settle into the city.",
                    "activities": ["waterfront walk"],
                    "duration": "3-4 hrs",
                }
            ],
            "warnings": ["Estimates are not live quotes."],
            "answer": "Your Alexandria plan is ready.",
        }
    )
    response = _build_structured_response(request, answer)

    assert response.trip_summary.destination == "Alexandria, Egypt"
    assert response.budget.total_egp == 14650
    assert response.budget.within_budget is True
    assert response.budget.utilization_percent == 81.4
    assert len(response.budget.breakdown) == 2
    assert response.budget.breakdown[0].category == "Stay"
    assert response.budget.breakdown[0].share_percent == 35
    assert len(response.itinerary) == 1
    assert response.itinerary[0].title == "Arrive"
    assert response.itinerary[0].activities == ["waterfront walk"]
    assert response.warnings == ["Estimates are not live quotes."]
    assert response.answer == "Your Alexandria plan is ready."


def test_create_trip_plan_endpoint_returns_structured_json():
    agent = Mock()
    agent.run.return_value = pyjson.dumps(
        {
            "trip_summary": {
                "destination": "Alexandria, Egypt",
                "start_date": "2026-11-12",
                "duration_days": 4,
                "travelers": 2,
                "travel_style": "Culture & history",
                "budget_amount_egp": 18000,
            },
            "budget": {
                "total_egp": 14650,
                "currency": "EGP",
                "budget_amount_egp": 18000,
                "within_budget": True,
                "breakdown": [
                    {"category": "Stay", "amount_egp": 5100, "note": "3 nights", "share_percent": 35},
                    {"category": "Food", "amount_egp": 3200, "note": "Local cafés", "share_percent": 22},
                ],
            },
            "itinerary": [
                {
                    "day": 1,
                    "title": "Arrive",
                    "summary": "Settle into the city.",
                    "activities": ["waterfront walk"],
                    "duration": "3-4 hrs",
                }
            ],
            "warnings": ["Estimates are not live quotes."],
            "answer": "Your Alexandria plan is ready.",
        }
    )

    client = _client_with_agent(agent)
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
            "thread_id": "endpoint-thread",
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["trip_summary"]["destination"] == "Alexandria, Egypt"
    assert data["budget"]["total_egp"] == 14650
    assert data["budget"]["within_budget"] is True
    assert data["budget"]["utilization_percent"] == 81.4
    assert len(data["itinerary"]) == 1
    assert data["itinerary"][0]["day"] == 1
    assert data["warnings"] == ["Estimates are not live quotes."]
    assert data["answer"] == "Your Alexandria plan is ready."
    assert data["thread_id"] == "endpoint-thread"
    agent.run.assert_called_once()


def test_create_trip_plan_endpoint_falls_back_on_plain_text():
    agent = Mock()
    agent.run.return_value = "Just a plain text answer with no structure."

    client = _client_with_agent(agent)
    resp = client.post(
        "/api/plan",
        json={
            "destination": "Cairo",
            "start_date": "2026-11-12",
            "duration_days": 3,
            "travelers": 1,
            "travel_style": "Relaxed getaway",
            "budget_amount": 10000,
            "budget_currency": "EGP",
            "thread_id": "fallback-thread",
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "Just a plain text answer with no structure."
    assert data["trip_summary"]["destination"] == "Cairo"
    assert data["budget"]["total_egp"] == 6000
    assert data["budget"]["breakdown_total_egp"] == 6000
    assert data["budget"]["within_budget"] is True
    assert len(data["itinerary"]) == 1
    assert any("baseline accommodation" in w.lower() for w in data["warnings"])
    assert not any("figures below are placeholders" in w.lower() for w in data["warnings"])


def test_create_trip_plan_endpoint_returns_502_on_agent_failure():
    agent = Mock()
    agent.run.side_effect = RuntimeError("model unavailable")

    client = _client_with_agent(agent)
    resp = client.post(
        "/api/plan",
        json={
            "destination": "Cairo",
            "start_date": "2026-11-12",
            "duration_days": 3,
            "travelers": 1,
            "travel_style": "Relaxed getaway",
            "budget_amount": 10000,
            "budget_currency": "EGP",
            "thread_id": "fail-thread",
        },
    )

    assert resp.status_code == 502
    assert "could not complete this request" in resp.json()["detail"]
