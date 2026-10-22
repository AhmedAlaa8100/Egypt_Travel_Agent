
"""Run the travel agent against the original evaluation scenarios."""

from dataclasses import dataclass
from typing import Any

from travel_agent.agent import TravelAgent
import os

from travel_agent.evaluation import (
    EVALUATION_SCENARIOS,
    evaluate_scenario,
)


@dataclass
class EvaluationResult:
    question: str
    answer: str
    passed: bool
    missing_values: list[str]


def evaluate_agent(
    agent: Any,
    scenarios: list[dict] | None = None,
) -> list[EvaluationResult]:
    """Evaluate an agent using expected values in each scenario."""
    scenarios = scenarios or EVALUATION_SCENARIOS
    results = []

    for scenario in scenarios:
        question = scenario["question"]
        answer = agent.run(question)
        evaluation = evaluate_scenario(answer, scenario)

        results.append(
            EvaluationResult(
                question=question,
                answer=answer,
                passed=evaluation["passed"],
                missing_values=(
                    evaluation["missing_values"]
                    + [
                        f"forbidden phrase: {phrase}"
                        for phrase in evaluation["forbidden_phrases_found"]
                    ]
                ),
            )
        )

    return results


def print_evaluation_report(results: list[EvaluationResult]) -> None:
    """Print a readable summary of evaluation results."""
    passed = sum(result.passed for result in results)

    for index, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] Scenario {index}: {result.question}")
        print(f"Answer: {result.answer}")

        if result.missing_values:
            print(f"Missing expected values: {result.missing_values}")

    print(f"\nOverall: {passed}/{len(results)} scenarios passed.")



def main() -> None:
    if os.getenv("RUN_LIVE_EVAL") != "1":
        raise SystemExit(
            "Live evaluation disabled to prevent accidental API usage. "
            "Set RUN_LIVE_EVAL=1 to enable it."
        )

    agent = TravelAgent()
    results = evaluate_agent(agent)
    print_evaluation_report(results)


if __name__ == "__main__":
    main()
