from datetime import date

from travel_agent.api import TripPlanRequest, build_agent_prompt


def test_build_agent_prompt_includes_form_details_and_price_guardrails():
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
