"""Evaluator module — scores an agent against a benchmark task set."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result of evaluating an agent on a task set."""
    total: int = 0
    correct: int = 0
    failures: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def accuracy_pct(self) -> str:
        return f"{self.accuracy * 100:.0f}%"


class Evaluator:
    """Evaluates an agent's accuracy on a list of (question, expected_answer) tasks."""

    def evaluate(self, agent, tasks: list[list[str]]) -> EvalResult:
        """Run the agent on every task and compute accuracy.

        Args:
            agent: An Agent instance with a .run(question) method.
            tasks: List of [question, expected_answer] pairs.

        Returns:
            An EvalResult with score and recorded failures.
        """
        result = EvalResult(total=len(tasks))

        for question, expected in tasks:
            try:
                response = agent.run(question)
            except Exception as exc:
                result.failures.append({
                    "question": question,
                    "expected": expected,
                    "response": f"[ERROR] {exc}",
                })
                continue

            if self._check_answer(response, expected):
                result.correct += 1
            else:
                result.failures.append({
                    "question": question,
                    "expected": expected,
                    "response": response.strip(),
                })

        return result

    @staticmethod
    def _check_answer(response: str, expected: str) -> bool:
        """Check if the expected answer appears in the response (case-insensitive)."""
        return expected.strip().lower() in response.strip().lower()
