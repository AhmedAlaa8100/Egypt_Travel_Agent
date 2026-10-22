
"""Evaluation scenarios and helpers for the travel agent."""

import re
from typing import Any


def normalize_response(response: Any) -> str:
    """Normalize response text for consistent evaluation."""
    return re.sub(
        r"(?<=\d)[,\s\u00a0\u202f](?=\d)",
        "",
        str(response).lower(),
    )


EVALUATION_SCENARIOS = [
    {
        "question": "How much will 3 nights in Aswan cost for 2 travellers?",
        "expected_values": ["7500"],
    },
    {
        "question": "Estimate a 4 night hotel stay in Cairo for 3 travellers.",
        "expected_values": ["24000"],
    },
    {
        "question": "Convert 100 USD to EGP.",
        "expected_values": ["4800"],
    },
    {
        "question": "Where should I go in Egypt for a warm winter trip?",
        "expected_values": ["alexandria"],
        "required_phrases": ["alexandria"],
        "forbidden_phrases": [],
    },
    {
        "question": (
            "How many nights does a Nile cruise between Luxor and Aswan take?"
        ),
        "expected_values": ["3", "4"],
    },
    {
        "question": (
            "How much is 2 nights in Alexandria for 2 travellers, "
            "and what is that in USD?"
        ),
        "expected_values": ["5600"],
        "alternative_values": ["116", "117"],
    },
]


def check_expected_values(
    response: Any,
    expected_values: list[str],
) -> bool:
    """Check whether every expected value appears in a response."""
    normalized = normalize_response(response)
    return all(
        normalize_response(value) in normalized
        for value in expected_values
    )


def evaluate_scenario(
    response: Any,
    scenario: dict,
) -> dict:
    """Evaluate expected values and optional phrase constraints."""
    normalized = normalize_response(response)
    missing = [
        value
        for value in scenario.get("expected_values", [])
        if normalize_response(value) not in normalized
    ]

    alternatives = scenario.get("alternative_values", [])
    if alternatives and not any(
        normalize_response(value) in normalized
        for value in alternatives
    ):
        missing.append(f"one of: {', '.join(alternatives)}")

    missing.extend(
        f"required phrase: {phrase}"
        for phrase in scenario.get("required_phrases", [])
        if normalize_response(phrase) not in normalized
    )

    forbidden_found = [
        phrase
        for phrase in scenario.get("forbidden_phrases", [])
        if normalize_response(phrase) in normalized
    ]

    return {
        "passed": not missing and not forbidden_found,
        "missing_values": missing,
        "forbidden_phrases_found": forbidden_found,
    }
