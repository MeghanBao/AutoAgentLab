"""Tests for PopulationLoop — no LLM calls (evaluator and mutator are mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoagentlab.agent import Agent
from autoagentlab.evaluator import EvalResult
from autoagentlab.population import PopulationLoop
from autoagentlab.scorer import ObjectiveWeights


def _make_agent(prompt: str = "test prompt") -> Agent:
    return Agent(prompt)


def _make_result(accuracy: float) -> EvalResult:
    r = EvalResult(total=2, correct=round(accuracy * 2))
    return r


# ── __init__ validation ─────────────────────────────────────────────

def test_population_size_must_be_at_least_2():
    with pytest.raises(ValueError, match="population_size must be >= 2"):
        PopulationLoop(_make_agent(), [["Q", "A"]], population_size=1)


def test_default_elite_size_is_half():
    loop = PopulationLoop(_make_agent(), [["Q", "A"]], population_size=4)
    assert loop.elite_size == 2


def test_custom_elite_size():
    loop = PopulationLoop(_make_agent(), [["Q", "A"]], population_size=6, elite_size=1)
    assert loop.elite_size == 1


def test_population_seeded_correctly():
    agent = _make_agent()
    loop = PopulationLoop(agent, [["Q", "A"]], population_size=3)
    assert len(loop.population) == 3
    # Index-0 is the original; others are clones with the same prompt
    assert loop.population[0] is agent
    for clone in loop.population[1:]:
        assert clone.prompt == agent.prompt
        assert clone.agent_id != agent.agent_id


# ── run() with mocked evaluator ─────────────────────────────────────

def _patched_loop(population_size: int = 2, iterations: int = 1) -> PopulationLoop:
    """Return a PopulationLoop with evaluator, researcher, and mutator mocked."""
    agent = _make_agent()
    tasks = [["Q1", "A"], ["Q2", "B"]]
    loop = PopulationLoop(agent, tasks, population_size=population_size, max_iterations=iterations)

    # Mock evaluator to return 80% accuracy for every agent
    loop.evaluator.evaluate = MagicMock(return_value=_make_result(0.8))
    # Mock researcher and mutator to return predictable strings
    loop.researcher.analyze = MagicMock(return_value="suggestion")
    loop.mutator.improve = MagicMock(return_value="improved prompt")

    return loop


def test_run_returns_history():
    loop = _patched_loop()
    history = loop.run()
    assert len(history) == 1              # 1 iteration
    assert len(history[0]) == 2          # 2 agents


def test_run_records_have_correct_ranks():
    loop = _patched_loop()
    records = loop.run()[0]
    assert records[0].rank == 1
    assert records[1].rank == 2


def test_best_agent_returns_population_zero():
    loop = _patched_loop()
    loop.run()
    assert loop.best_agent() is loop.population[0]


def test_elite_agents_preserved():
    """After evolution, the population should have population_size members."""
    loop = _patched_loop(population_size=4, iterations=2)
    loop.run()
    assert len(loop.population) == 4


def test_mutations_generated():
    """Mutator should be called for non-elite slots."""
    loop = _patched_loop(population_size=4, iterations=1)
    loop.run()
    # elite_size=2, slots=2 → mutator called 2 times
    assert loop.mutator.improve.call_count == 2


def test_multi_objective_weights_passed():
    weights = ObjectiveWeights(accuracy=1.0, latency=0.5)
    agent = _make_agent()
    loop = PopulationLoop(
        agent, [["Q", "A"]], population_size=2, weights=weights
    )
    assert loop.weights.latency == 0.5
