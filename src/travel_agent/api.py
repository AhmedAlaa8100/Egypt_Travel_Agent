"""FastAPI bridge between the Roamwise HTML UI and the existing TravelAgent."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from travel_agent.agent import TravelAgent
from travel_agent.tools.travel_tools import estimate_trip_cost

logger = logging.getLogger(__name__)
app = FastAPI(title="Roamwise Travel Agent API", version="1.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
_agent: TravelAgent | None = None
_agent_lock = Lock()
_run_lock = Lock()

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
# Models occasionally emit a stray quote after a number, for example
# {"day": 3", "title": ...}. Repairing these small mistakes is safer than
# dropping a good plan and showing the raw JSON blob to the user.
_STRAY_QUOTE_AFTER_NUMBER = re.compile(r'(:\s*-?\d+(?:\.\d+)?)\s*"')
_STRAY_QUOTE_AFTER_LITERAL = re.compile(
    r'(:\s*(?:true|false|null))\s*"', re.IGNORECASE
)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
)
_ANSWER_IN_BROKEN_JSON = re.compile(
    r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', re.IGNORECASE
)
# A few line items are really "the rest of the budget" rather than a cost.
# They inflate the total to exactly the user's cap, so they are dropped.
_PLACEHOLDER_CATEGORIES = {
    "other",
    "others",
    "remaining",
    "remainder",
    "unallocated",
    "placeholder",
    "buffer",
    "contingency",
    "reserve",
}
_PLACEHOLDER_NOTE = re.compile(
    r"\b(?:placeholder|remaining budget|budget remainder|unallocated|"
    r"not itemi[sz]ed|not included|missing expenses?|buffer for)\b",
    re.IGNORECASE,
)
_TEXT_TRANSLATION = str.maketrans(
    {
        "\u2011": "-",  # non-breaking hyphen
        "\u2013": "-",  # en dash inside words such as day-by-day
        "\u00a0": " ",
        "\u2007": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)
_MAX_AGENT_ATTEMPTS = 3


class TripPlanRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=160)
    start_date: date | None = None
    duration_days: int = Field(ge=1, le=30)
    travelers: int = Field(ge=1, le=10)
    travel_style: str = Field(min_length=2, max_length=80)
    budget_amount: int = Field(gt=0, le=100_000_000)
    budget_currency: Literal["EGP"] = "EGP"
    thread_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_-]+$")


class CostEntry(BaseModel):
    category: str
    amount_egp: int = Field(ge=0)
    note: str = ""
    share_percent: int = Field(default=0, ge=0, le=100)


class BudgetCard(BaseModel):
    total_egp: int = Field(ge=0)
    currency: str = "EGP"
    budget_amount_egp: int = Field(ge=0)
    within_budget: bool = True
    utilization_percent: float = Field(default=0.0, ge=0.0)
    breakdown: list[CostEntry] = []
    breakdown_total_egp: int = Field(default=0, ge=0)


class ItineraryDay(BaseModel):
    day: int = Field(ge=1)
    title: str
    summary: str
    activities: list[str] = []
    duration: str | None = None


class TripSummary(BaseModel):
    destination: str
    start_date: str | None = None
    duration_days: int = Field(ge=1)
    travelers: int = Field(ge=1)
    travel_style: str
    budget_amount_egp: int = Field(ge=0)


class TripPlanResponse(BaseModel):
    trip_summary: TripSummary
    budget: BudgetCard
    itinerary: list[ItineraryDay] = []
    warnings: list[str] = []
    answer: str = ""
    thread_id: str


def get_agent() -> TravelAgent:
    """Create the agent only when a planning request arrives."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = TravelAgent()
    return _agent


def build_agent_prompt(request: TripPlanRequest) -> str:
    """Turn the form fields into a clear request for the existing agent."""
    date_text = request.start_date.isoformat() if request.start_date else "not specified"
    return (
        "Create a practical travel plan using the travel tools and knowledge base "
        "available to you. The destination is in Egypt unless the destination says otherwise. "
        "Do not invent live prices or claim bookings have been made. Clearly label estimates "
        "and mention missing information instead of guessing.\n\n"
        f"Destination: {request.destination}\n"
        f"Departure date: {date_text}\n"
        f"Duration: {request.duration_days} days\n"
        f"Travelers: {request.travelers}\n"
        f"Travel style: {request.travel_style}\n"
        f"Total budget: {request.budget_amount:,} {request.budget_currency}\n\n"
        "Suggest a day-by-day itinerary and explain whether the budget seems plausible. "
        "Use available tools when useful. If your available tools cannot calculate a complete "
        "trip budget, say so explicitly and do not present a partial cost as the full trip total.\n\n"
        "The budget object is required. Use estimate_trip_cost for accommodation when the "
        "destination is supported. Put every cost you can estimate in breakdown, and set "
        "total_egp to the sum of those line items. The user's budget is a cap, not a cost: "
        "never copy it into total_egp. If some costs are unknown, omit them and list them in "
        "warnings. If you cannot estimate any cost, set total_egp to 0 and breakdown to [] "
        "and explain why in warnings.\n\n"
        "Return your final answer as a JSON object with exactly these keys and no extra text:\n"
        "- trip_summary: {destination, start_date, duration_days, travelers, travel_style, budget_amount_egp}\n"
        "- budget: {total_egp, currency, budget_amount_egp, within_budget, breakdown: [{category, amount_egp, note, share_percent}]}\n"
        "- itinerary: [{day, title, summary, activities: [strings], duration}]\n"
        "- warnings: [strings about estimates, missing info, or budget concerns]\n"
        "- answer: a short human-readable summary of the plan\n"
        "Use plain numbers for amounts (EGP) and whole numbers for share_percent. "
        "Make the breakdown amounts add up to total_egp. The JSON must be valid: "
        "no trailing commas, no unescaped quotes, and no markdown code fences."
    )


def _brace_objects(text: str, limit: int = 25) -> list[str]:
    """Return top-level ``{...}`` substrings, ignoring string state.

    A malformed quote can confuse a string-aware scanner, so this simple
    depth counter is used as one of several candidate extractors. It is
    only ever fed to ``json.loads`` afterwards.
    """
    objects: list[str] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : index + 1])
                    start = None
                    if len(objects) >= limit:
                        break
    return objects


def _close_truncated_json(candidate: str) -> str:
    """Close strings and containers left open by a truncated response."""
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack:
            stack.pop()

    repaired = candidate
    if in_string:
        repaired += '"'
    repaired += "".join(reversed(stack))
    return repaired


def _repair_json(candidate: str) -> object | None:
    """Make a best effort to decode slightly malformed model JSON."""
    repaired = candidate.translate(_QUOTE_TRANSLATION)
    repaired = repaired.replace("\ufeff", "").strip()
    repaired = _TRAILING_COMMA.sub(r"\1", repaired)
    repaired = _STRAY_QUOTE_AFTER_NUMBER.sub(r"\1", repaired)
    repaired = _STRAY_QUOTE_AFTER_LITERAL.sub(r"\1", repaired)
    repaired = re.sub(r"(:\s*)True\b", r"\1true", repaired)
    repaired = re.sub(r"(:\s*)False\b", r"\1false", repaired)
    repaired = re.sub(r"(:\s*)None\b", r"\1null", repaired)
    repaired = re.sub(r"(:\s*)NaN\b", r"\1null", repaired)
    repaired = re.sub(r"(:\s*)-?Infinity\b", r"\1null", repaired)
    repaired = _close_truncated_json(repaired)

    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _json_candidates(text: str) -> list[str]:
    """Collect possible JSON fragments from a model response."""
    candidates = [
        match.group(1).strip() for match in _JSON_FENCE.finditer(text)
    ]

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    candidates.extend(_brace_objects(stripped))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _pick_json_object(data: object, require_plan: bool = False) -> dict | None:
    """Accept a plan object, or the first object inside a list."""
    candidates: list[dict] = []
    if isinstance(data, dict):
        candidates = [data]
    elif isinstance(data, list):
        candidates = [item for item in data if isinstance(item, dict)]

    if require_plan:
        for item in candidates:
            if {"trip_summary", "budget", "itinerary"} & set(item):
                return item
        return None

    return candidates[0] if candidates else None


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of the agent answer.

    Handles code fences, prose around the object, and small syntax slips
    such as trailing commas or a stray quote after a number. Strict parsing
    is attempted first so a valid object is never altered.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    candidates = _json_candidates(text)

    for require_plan in (True, False):
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            picked = _pick_json_object(parsed, require_plan=require_plan)
            if picked is not None:
                return picked

    for require_plan in (True, False):
        for candidate in candidates:
            picked = _pick_json_object(
                _repair_json(candidate), require_plan=require_plan
            )
            if picked is not None:
                return picked

    return None


def _coerce_int(value, default: int = 0) -> int:
    """Convert model output such as ``"1,500 EGP"`` to an integer."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9eE+\-.]", "", value.replace(",", "").strip())
        try:
            number = float(cleaned)
        except (TypeError, ValueError):
            return default
        return int(number) if math.isfinite(number) else default
    return default


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


_PLAN_KEYS = {
    "trip_summary",
    "budget",
    "itinerary",
    "warnings",
    "answer",
    "destination",
    "total_egp",
    "breakdown",
}



def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_items(value) -> list:
    """Treat a single object as a one-item list, and reject other types."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return []


def _clean_text(value, limit: int = 2000) -> str:
    """Return short, safe text without ever dumping nested JSON."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, (int, float)):
        text = str(value)
    else:
        return ""

    text = text.replace("\x00", "").translate(_TEXT_TRANSLATION).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _is_placeholder_entry(category: str, note: str) -> bool:
    """Return True for rows that only restate the user's budget cap."""
    key = re.sub(r"[^a-z ]", "", category.lower()).strip()
    generic = (
        key in _PLACEHOLDER_CATEGORIES
        or key.startswith("other")
        or "remaining" in key
        or "placeholder" in key
        or "unallocated" in key
    )
    return generic and bool(_PLACEHOLDER_NOTE.search(note))


def _response_needs_retry(response: TripPlanResponse) -> bool:
    """A plan with no cost estimate at all is worth one more agent attempt."""
    return response.budget.total_egp <= 0 and not response.budget.breakdown


def _apply_baseline_estimate(
    request: TripPlanRequest, response: TripPlanResponse
) -> TripPlanResponse:
    """Guarantee a transparent baseline cost when the model keeps failing.

    Only accommodation can be estimated deterministically, so the fallback is
    deliberately narrow and says so in a warning. Unsupported destinations are
    left untouched rather than given an invented number.
    """
    nights = max(1, request.duration_days - 1)
    try:
        stay = estimate_trip_cost(
            request.destination, nights, request.travelers
        )
    except (TypeError, ValueError):
        return response

    amount = max(0, int(stay["total_egp"]))
    if amount <= 0:
        return response

    rooms = int(stay["rooms"])
    note = (
        f"{nights} {'night' if nights == 1 else 'nights'}, "
        f"{rooms} {'room' if rooms == 1 else 'rooms'} · baseline estimate"
    )
    response.budget = BudgetCard(
        total_egp=amount,
        currency=request.budget_currency,
        budget_amount_egp=request.budget_amount,
        within_budget=amount <= request.budget_amount,
        utilization_percent=(
            round(amount / request.budget_amount * 100, 1)
            if request.budget_amount
            else 0.0
        ),
        breakdown=[
            CostEntry(
                category="Accommodation",
                amount_egp=amount,
                note=note,
                share_percent=100,
            )
        ],
        breakdown_total_egp=amount,
    )

    warning = (
        "The agent did not return usable cost figures after several attempts. "
        "A baseline accommodation estimate is shown instead; meals, transport, "
        "flights, and activities are not included."
    )
    obsolete = (
        "no reliable cost estimate is available",
        "figures below are placeholders",
        "did not return a numeric total cost",
    )
    response.warnings = [
        item
        for item in response.warnings
        if not any(marker in item.lower() for marker in obsolete)
    ]
    if warning not in response.warnings:
        response.warnings = [*response.warnings, warning]

    answer = (response.answer or "").lower()
    if not answer or "empty response" in answer or "could not be read" in answer:
        response.answer = (
            f"A {request.duration_days}-day plan for "
            f"{request.travelers} "
            f"{'traveller' if request.travelers == 1 else 'travellers'} in "
            f"{request.destination}. A baseline accommodation estimate of "
            f"EGP {amount:,} is shown. Meals, transport, flights, and "
            "activities are not included."
        )
    return response


def _looks_like_json_text(text: str) -> bool:
    stripped = text.lstrip()
    return (
        stripped.startswith(("{", "["))
        or '"trip_summary"' in text
        or '"itinerary"' in text
    )


def _summary_answer(
    destination: str,
    days: int,
    travelers: int,
    total_egp: int,
    budget_egp: int,
    within_budget: bool,
) -> str:
    travellers_label = "traveller" if travelers == 1 else "travellers"
    if total_egp > 0:
        budget_position = (
            f"within your EGP {budget_egp:,} budget"
            if within_budget
            else f"over your EGP {budget_egp:,} budget"
        )
        return (
            f"A {days}-day plan for {travelers} {travellers_label} in "
            f"{destination}, with an estimated total of EGP {total_egp:,}, "
            f"which is {budget_position}."
        )
    return (
        f"A {days}-day plan for {travelers} {travellers_label} in "
        f"{destination}. The agent did not return a reliable total cost, "
        "so treat the figures as incomplete."
    )


def _readable_answer(
    answer_text: str,
    data: dict | None,
    destination: str,
    days: int,
    travelers: int,
    total_egp: int,
    budget_egp: int,
    within_budget: bool,
) -> str:
    """Return a human-readable answer, never a raw JSON payload."""
    if data is not None:
        answer = _clean_text(data.get("answer"), limit=6000)
        if answer:
            return answer
        return _summary_answer(
            destination, days, travelers, total_egp, budget_egp, within_budget
        )

    raw = _clean_text(answer_text, limit=6000)
    if not raw:
        return "The agent returned an empty response. Please try again."

    # The answer may be broken JSON. Try to salvage its answer field so the
    # user still gets the written summary instead of a JSON blob.
    if _looks_like_json_text(raw):
        match = _ANSWER_IN_BROKEN_JSON.search(answer_text)
        if match:
            try:
                salvaged = json.loads(f'"{match.group(1)}"')
            except (json.JSONDecodeError, TypeError, ValueError):
                salvaged = match.group(1)
            salvaged = _clean_text(salvaged, limit=6000)
            if salvaged:
                return salvaged
        return (
            "The agent returned a response that could not be read as a plan. "
            "Please try building the plan again."
        )

    return raw


def _fallback_response(
    request: TripPlanRequest, answer_text: str
) -> TripPlanResponse:
    """Last-resort response if unexpected data slips past the helpers."""
    return TripPlanResponse(
        trip_summary=TripSummary(
            destination=request.destination,
            start_date=(
                request.start_date.isoformat() if request.start_date else None
            ),
            duration_days=request.duration_days,
            travelers=request.travelers,
            travel_style=request.travel_style,
            budget_amount_egp=request.budget_amount,
        ),
        budget=BudgetCard(
            total_egp=0,
            budget_amount_egp=request.budget_amount,
            within_budget=True,
            utilization_percent=0.0,
            breakdown=[],
            breakdown_total_egp=0,
        ),
        itinerary=[],
        warnings=[
            "The agent returned a response that could not be read as a plan."
        ],
        answer=_readable_answer(
            answer_text,
            None,
            request.destination,
            request.duration_days,
            request.travelers,
            0,
            request.budget_amount,
            True,
        ),
        thread_id=request.thread_id,
    )


def _build_structured_response(
    request: TripPlanRequest, answer_text: str
) -> TripPlanResponse:
    """Turn the agent's answer into a structured response, with a safe fallback."""
    if not isinstance(answer_text, str):
        answer_text = str(answer_text or "")

    try:
        data = _extract_json(answer_text)
        raw = (
            data
            if isinstance(data, dict) and (_PLAN_KEYS & set(data))
            else None
        )
        summary_raw = _as_dict(raw.get("trip_summary")) if raw else {}

        destination = (
            _clean_text(summary_raw.get("destination"), limit=160)
            or request.destination
        )
        start_date = (
            request.start_date.isoformat()
            if request.start_date
            else _clean_text(summary_raw.get("start_date"), limit=40) or None
        )
        travel_style = (
            _clean_text(summary_raw.get("travel_style"), limit=80)
            or request.travel_style
        )

        trip_summary = TripSummary(
            destination=destination,
            start_date=start_date,
            duration_days=request.duration_days,
            travelers=request.travelers,
            travel_style=travel_style,
            budget_amount_egp=request.budget_amount,
        )

        budget_raw = _as_dict(raw.get("budget")) if raw else {}
        prepared_entries = []
        placeholder_total = 0
        for entry in _as_items(budget_raw.get("breakdown")):
            if not isinstance(entry, dict):
                continue
            prepared = {
                "category": (
                    _clean_text(entry.get("category"), limit=80) or "Other"
                ),
                "amount_egp": max(
                    0, _coerce_int(entry.get("amount_egp", 0))
                ),
                "note": _clean_text(entry.get("note"), limit=400),
                "share_percent": _coerce_int(
                    entry.get("share_percent"), 0
                ),
            }
            if _is_placeholder_entry(
                prepared["category"], prepared["note"]
            ):
                placeholder_total += prepared["amount_egp"]
                continue
            prepared_entries.append(prepared)

        breakdown_total = sum(
            entry["amount_egp"] for entry in prepared_entries
        )
        breakdown = [
            CostEntry(
                category=entry["category"],
                amount_egp=entry["amount_egp"],
                note=entry["note"],
                share_percent=max(
                    0,
                    min(
                        100,
                        entry["share_percent"]
                        or (
                            round(entry["amount_egp"] / breakdown_total * 100)
                            if breakdown_total
                            else 0
                        ),
                    ),
                ),
            )
            for entry in prepared_entries
        ]

        total_egp = max(0, _coerce_int(budget_raw.get("total_egp", 0)))
        budget_notes: list[str] = []
        if placeholder_total:
            total_egp = max(0, total_egp - placeholder_total)
            budget_notes.append(
                f"A budget placeholder of EGP {placeholder_total:,} was "
                "excluded from the estimate because it repeated the trip "
                "budget rather than describing a cost."
            )
        if total_egp == 0 and breakdown_total > 0:
            total_egp = breakdown_total
        if breakdown_total and total_egp < breakdown_total:
            budget_notes.append(
                f"The agent's total was lower than its itemised costs, so "
                f"the itemised total of EGP {breakdown_total:,} is shown "
                "instead."
            )
            total_egp = breakdown_total

        within_budget = total_egp <= request.budget_amount
        budget = BudgetCard(
            total_egp=total_egp,
            currency=request.budget_currency,
            budget_amount_egp=request.budget_amount,
            within_budget=within_budget,
            utilization_percent=(
                round(total_egp / request.budget_amount * 100, 1)
                if request.budget_amount
                else 0.0
            ),
            breakdown=breakdown,
            breakdown_total_egp=breakdown_total,
        )

        itinerary: list[ItineraryDay] = []
        raw_itinerary = _as_items(raw.get("itinerary")) if raw else []
        for index, item in enumerate(raw_itinerary):
            if isinstance(item, str):
                day = index + 1
                title = f"Day {day}"
                summary = _clean_text(item, limit=2000)
                activities: list[str] = []
                duration = None
            elif isinstance(item, dict):
                day = _coerce_int(item.get("day"), index + 1)
                if day < 1:
                    day = index + 1
                title = (
                    _clean_text(item.get("title"), limit=200) or f"Day {day}"
                )
                summary = _clean_text(item.get("summary"), limit=2000)
                raw_activities = item.get("activities")
                if isinstance(raw_activities, str):
                    raw_activities = re.split(r"[;\n]", raw_activities)
                activities = []
                for activity in _as_list(raw_activities)[:20]:
                    cleaned_activity = _clean_text(activity, limit=300)
                    if cleaned_activity:
                        activities.append(cleaned_activity)
                duration = _clean_text(item.get("duration"), limit=60) or None
            else:
                continue

            itinerary.append(
                ItineraryDay(
                    day=day,
                    title=title,
                    summary=summary,
                    activities=activities,
                    duration=duration,
                )
            )

        itinerary.sort(key=lambda entry: entry.day)
        for position, entry in enumerate(itinerary, start=1):
            entry.day = position

        answer = _readable_answer(
            answer_text,
            raw,
            destination,
            request.duration_days,
            request.travelers,
            total_egp,
            request.budget_amount,
            within_budget,
        )

        if not itinerary and raw is None:
            itinerary.append(
                ItineraryDay(
                    day=1,
                    title="Plan details",
                    summary=_clean_text(answer, limit=400),
                )
            )

        warnings: list[str] = []
        if raw is not None:
            raw_warnings = raw.get("warnings")
            if isinstance(raw_warnings, str):
                raw_warnings = [raw_warnings]
            for warning in _as_list(raw_warnings):
                cleaned_warning = _clean_text(warning, limit=800)
                if cleaned_warning and cleaned_warning not in warnings:
                    warnings.append(cleaned_warning)
            for note in budget_notes:
                if note not in warnings:
                    warnings.append(note)
            if total_egp == 0:
                warnings.append(
                    "The agent did not return a numeric total cost; the "
                    "budget estimate is incomplete."
                )
        if raw is None:
            warnings.insert(
                0,
                "The agent returned an unstructured response; no reliable "
                "cost estimate is available.",
            )

        return TripPlanResponse(
            trip_summary=trip_summary,
            budget=budget,
            itinerary=itinerary,
            warnings=warnings,
            answer=answer,
            thread_id=request.thread_id,
        )
    except Exception:  # pragma: no cover - defensive guard for model output
        logger.exception("Could not structure the agent response")
        return _fallback_response(request, answer_text)



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/plan", response_model=TripPlanResponse)
def create_trip_plan(request: TripPlanRequest) -> TripPlanResponse:
    response: TripPlanResponse | None = None
    last_error: Exception | None = None

    for attempt in range(_MAX_AGENT_ATTEMPTS):
        # A new thread per submission avoids stale checkpoint history. Retries
        # get their own thread so a failed attempt cannot poison the next one.
        thread_id = request.thread_id
        if attempt:
            suffix = f"-retry{attempt}"
            thread_id = f"{request.thread_id[: 120 - len(suffix)]}{suffix}"

        try:
            # The agent owns a shared graph/checkpointer. Serialise requests so
            # two browser tabs cannot interleave graph state for one process.
            with _run_lock:
                answer = get_agent().run(
                    build_agent_prompt(request),
                    thread_id=thread_id,
                )
        except Exception as exc:  # transient provider or tool failure
            last_error = exc
            logger.warning(
                "Travel agent attempt %s/%s failed: %s",
                attempt + 1,
                _MAX_AGENT_ATTEMPTS,
                exc,
            )
            continue

        response = _build_structured_response(request, str(answer))
        if not _response_needs_retry(response):
            break
        logger.warning(
            "Travel agent attempt %s/%s returned no usable estimate; retrying",
            attempt + 1,
            _MAX_AGENT_ATTEMPTS,
        )

    if response is None:
        logger.exception(
            "Travel agent request failed after %s attempts",
            _MAX_AGENT_ATTEMPTS,
            exc_info=last_error,
        )
        # Keep internal errors and provider details out of the browser response.
        raise HTTPException(
            status_code=502,
            detail="The travel agent could not complete this request. Check the server terminal and API configuration, then try again.",
        ) from last_error

    if _response_needs_retry(response):
        response = _apply_baseline_estimate(request, response)

    response.thread_id = request.thread_id
    return response
