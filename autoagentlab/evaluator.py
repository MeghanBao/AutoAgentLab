"""Evaluator module — scores an agent against a benchmark task set."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result of evaluating an agent on a task set."""

    total: int = 0
    correct: int = 0
    failures: list[dict] = field(default_factory=list)
    total_latency_s: float = 0.0  # sum of wall-clock seconds for all run() calls
    total_tokens: int = 0          # sum of agent._last_usage across all tasks

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def accuracy_pct(self) -> str:
        return f"{self.accuracy * 100:.0f}%"

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.total if self.total > 0 else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate (gpt-4o-mini blended rate ~$0.38/1M tokens)."""
        return self.total_tokens * 0.00000038


class Evaluator:
    """Evaluates an agent's accuracy on a list of (question, expected_answer) tasks."""

    def evaluate(self, agent, tasks: list[list[str]]) -> EvalResult:
        """Run the agent on every task and compute accuracy.

        Also accumulates wall-clock latency and token usage from
        *agent._last_usage* after each call for multi-objective scoring.

        Args:
            agent: An Agent instance with a .run(question) method.
            tasks: List of [question, expected_answer] pairs.

        Returns:
            An EvalResult with score, failures, latency, and token usage.
        """
        result = EvalResult(total=len(tasks))

        for question, expected in tasks:
            t0 = time.perf_counter()
            try:
                response = agent.run(question)
            except Exception as exc:
                result.failures.append({
                    "question": question,
                    "expected": expected,
                    "response": f"[ERROR] {exc}",
                })
                result.total_latency_s += time.perf_counter() - t0
                continue

            result.total_latency_s += time.perf_counter() - t0
            result.total_tokens += getattr(agent, "_last_usage", 0)

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
