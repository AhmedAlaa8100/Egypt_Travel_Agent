
"""Build the LangChain tools available to the travel agent."""

from langchain_core.tools import BaseTool, tool

from travel_agent.tools.preference_tools import PreferenceTools
from travel_agent.tools.travel_tools import (
    convert_currency as convert_currency_impl,
    estimate_trip_cost as estimate_trip_cost_impl,
)


def create_tools(preferences: PreferenceTools) -> list[BaseTool]:
    """Create the tools, injecting the preference memory dependency."""

    @tool
    def estimate_trip_cost(
        destination: str,
        nights: int,
        travellers: int = 1,
    ) -> dict:
        """Estimate accommodation costs in EGP for an Egyptian destination."""
        return estimate_trip_cost_impl(
            destination=destination,
            nights=nights,
            travellers=travellers,
        )

    @tool
    def convert_currency(
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> dict:
        """Convert an amount between EGP, USD, and EUR using configured prototype rates."""
        return convert_currency_impl(
            amount=amount,
            from_currency=from_currency,
            to_currency=to_currency,
        )

    @tool
    def save_preference(fact: str) -> dict:
        """Save a traveller fact or preference for future reference."""
        return preferences.save_preference(fact)

    @tool
    def recall_preferences(query: str) -> dict:
        """Retrieve previously saved traveller preferences relevant to a query."""
        return preferences.recall_preferences(query)

    return [
        estimate_trip_cost,
        convert_currency,
        save_preference,
        recall_preferences,
    ]
