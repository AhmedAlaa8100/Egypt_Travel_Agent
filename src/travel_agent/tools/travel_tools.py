
"""Travel-related tools used by the travel agent."""

# Prototype rates, expressed in EGP per room per night.
NIGHTLY_RATE_EGP = {
    "aswan": 2500,
    "cairo": 3000,
    "alexandria": 2800,
    "luxor": 2600,
}

# Placeholder exchange rates: amount of EGP per unit of currency.
# These are NOT live exchange rates.
FX_TO_EGP = {
    "EGP": 1.0,
    "USD": 48.0,
    "EUR": 52.0,
}


def estimate_trip_cost(destination, nights, travellers=1):
    """Estimate hotel cost in EGP for a stay in an Egyptian city."""
    dest = str(destination).strip().lower()

    if dest not in NIGHTLY_RATE_EGP:
        raise ValueError(
            f"Unknown destination '{destination}'. "
            f"Choose from: {sorted(NIGHTLY_RATE_EGP)}"
        )

    nights = int(nights)
    travellers = int(travellers)

    if not 1 <= nights <= 60 or not 1 <= travellers <= 10:
        raise ValueError("nights must be 1-60 and travellers 1-10")

    # Assume up to two travellers share one room.
    rooms = (travellers + 1) // 2
    total = NIGHTLY_RATE_EGP[dest] * nights * rooms

    return {
        "destination": dest,
        "nights": nights,
        "rooms": rooms,
        "total_egp": total,
        "total_text": f"{total} EGP",
    }


def convert_currency(amount, from_currency, to_currency):
    """Convert between EGP, USD and EUR using prototype rates."""
    src = str(from_currency).upper()
    dst = str(to_currency).upper()

    if src not in FX_TO_EGP or dst not in FX_TO_EGP:
        raise ValueError(
            f"Supported currencies: {sorted(FX_TO_EGP)}"
        )

    amount = float(amount)

    if amount < 0:
        raise ValueError("amount cannot be negative")

    result = amount * FX_TO_EGP[src] / FX_TO_EGP[dst]

    return {
        "amount": amount,
        "from": src,
        "to": dst,
        "result": round(result, 2),
    }
