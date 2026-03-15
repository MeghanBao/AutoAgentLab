"""Evaluator module — scores an agent against a benchmark task set."""

from __future__ import annotations

import re
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


# ── Answer-checking helpers ─────────────────────────────────────────

# Finds all numbers in a string, including negatives and decimals
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)


def _parse_number(s: str) -> float:
    """Parse a single string to float; raises ValueError if not numeric."""
    cleaned = s.strip().replace(",", "").replace(" ", "")
    return float(cleaned)


def _numbers_close(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.5) -> bool:
    """Return True if a and b are within 1% relative or 0.5 absolute tolerance."""
    if b == 0:
        return abs(a) <= abs_tol
    return abs(a - b) / abs(b) <= rel_tol or abs(a - b) <= abs_tol


def _match_single(response: str, expected: str) -> bool:
    """Check whether *response* satisfies a single expected-answer string.

    Strategy:
    - If *expected* is purely numeric: use only numeric tolerance matching
      (avoids false positives like "8" matching "80" via substring).
    - Otherwise: use case-insensitive substring matching.
    """
    # 1. Numeric tolerance match for numeric expected values.
    #    Handles "3.14" ≈ "3.14159", "9,716" == "9716", "195.0" == "195", etc.
    try:
        exp_num = _parse_number(expected)
        # Strip commas from response so "9,716" is read as 9716
        resp_clean = response.replace(",", "")
        for m in _NUMBER_RE.finditer(resp_clean):
            try:
                if _numbers_close(float(m.group()), exp_num):
                    return True
            except ValueError:
                pass
        # expected IS numeric but nothing in response matched → False
        return False
    except ValueError:
        pass  # expected is not a plain number — fall through to text match

    # 2. Case-insensitive substring match for text answers.
    return expected.strip().lower() in response.strip().lower()


class Evaluator:
    """Evaluates an agent's accuracy on a list of (question, expected_answer) tasks.

    Expected-answer format in benchmark JSON:
        - Single string:        "Paris"
        - Pipe-separated alts:  "same|equal|neither"  (any one match counts)

    Answer checking (in order):
        1. Case-insensitive substring match.
        2. Numeric tolerance match (1% relative or ±0.5 absolute).
    """

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
        """Return True if *response* matches *expected*.

        Supports pipe-separated alternatives (any one match counts) and
        numeric tolerance matching in addition to substring matching.
        """
        alternatives = [a.strip() for a in expected.split("|")]
        return any(_match_single(response, alt) for alt in alternatives)
