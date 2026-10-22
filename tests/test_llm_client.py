import os

import pytest

from travel_agent.llm.client import LLMClient


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY is not set in the environment",
)
def test_llm_client_live_request():
    client = LLMClient()

    response = client.generate(
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Travel agent ready",
            }
        ]
    )

    assert response.choices
    assert response.choices[0].message.content
