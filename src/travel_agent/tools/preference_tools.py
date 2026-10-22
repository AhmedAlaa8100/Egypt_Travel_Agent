from travel_agent.memory.store import MemoryStore


class PreferenceTools:
    """Exposes memory operations as agent-callable tools."""

    def __init__(self, memory: MemoryStore | None = None) -> None:
        self._memory = memory or MemoryStore()

    def save_preference(self, fact: str) -> dict:
        """Save a fact or preference about the traveller."""

        if not isinstance(fact, str):
            raise TypeError("fact must be a string")

        fact = fact.strip()
        if len(fact) < 3:
            raise ValueError("fact must be a non-empty sentence")

        self._memory.remember(
            fact,
            metadata={"type": "preference"},
        )

        return {"saved": fact}

    def recall_preferences(self, query: str) -> dict:
        """Look up stored facts about the traveller."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")

        return {"memories": self._memory.recall(query, top_k=3)}
