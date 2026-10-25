import pytest

from travel_agent.tools.travel_tools import estimate_trip_cost


def test_estimate_accepts_a_city_with_a_country_suffix():
    estimate = estimate_trip_cost("Alexandria, Egypt", 3, 2)

    assert estimate["destination"] == "alexandria"
    assert estimate["total_egp"] == 8400


def test_estimate_normalises_case_and_whitespace():
    estimate = estimate_trip_cost("  CAIRO  ", 2, 1)

    assert estimate["destination"] == "cairo"
    assert estimate["total_egp"] == 6000


def test_estimate_rejects_an_unknown_destination():
    with pytest.raises(ValueError, match="Unknown destination"):
        estimate_trip_cost("Paris, France", 2, 1)
