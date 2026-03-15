"""Tests for multi-objective scoring — no LLM calls required."""

from autoagentlab.evaluator import EvalResult
from autoagentlab.scorer import ObjectiveWeights, composite_score


def _result(accuracy: float, latency_s: float = 0.0, tokens: int = 0) -> EvalResult:
    r = EvalResult(total=100, correct=int(accuracy * 100))
    r.total_latency_s = latency_s * 100   # avg * total tasks
    r.total_tokens = tokens
    return r


def test_accuracy_only_default_weights():
    weights = ObjectiveWeights()
    r = _result(0.75)
    assert abs(composite_score(r, weights) - 0.75) < 1e-9


def test_accuracy_only_explicit():
    weights = ObjectiveWeights(accuracy=1.0, latency=0.0, cost=0.0)
    r = _result(0.60)
    assert abs(composite_score(r, weights) - 0.60) < 1e-9


def test_latency_penalizes_slow():
    weights = ObjectiveWeights(accuracy=1.0, latency=1.0)
    fast = _result(0.8, latency_s=1.0)
    slow = _result(0.8, latency_s=5.0)
    assert composite_score(fast, weights) > composite_score(slow, weights)


def test_cost_penalizes_expensive():
    weights = ObjectiveWeights(accuracy=1.0, cost=1.0)
    cheap = _result(0.8, tokens=100)
    expensive = _result(0.8, tokens=100_000)
    assert composite_score(cheap, weights) > composite_score(expensive, weights)


def test_composite_fast_beats_accurate_under_latency_weight():
    """A faster agent can outscore a more accurate one when latency weight is high."""
    weights = ObjectiveWeights(accuracy=1.0, latency=5.0)
    slower_accurate = _result(0.9, latency_s=10.0)
    faster_less_accurate = _result(0.8, latency_s=0.5)
    # 0.9 - 5*10 = -49.1  vs  0.8 - 5*0.5 = -1.7
    assert composite_score(faster_less_accurate, weights) > composite_score(slower_accurate, weights)


def test_zero_latency_zero_tokens():
    weights = ObjectiveWeights(accuracy=1.0, latency=1.0, cost=1.0)
    r = _result(0.5, latency_s=0.0, tokens=0)
    assert composite_score(r, weights) == 0.5
