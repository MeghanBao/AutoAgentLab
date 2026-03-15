"""Multi-objective scoring for agent acceptance decisions."""

from __future__ import annotations

from dataclasses import dataclass

from autoagentlab.evaluator import EvalResult


@dataclass
class ObjectiveWeights:
    """Weights controlling which objectives matter in accept/reject decisions.

    accuracy:  higher is better (weight multiplied positively).
    latency:   lower is better (weight acts as a penalty on avg_latency_s).
    cost:      lower is better (weight acts as a penalty on cost in milli-dollars).
    """

    accuracy: float = 1.0
    latency: float = 0.0
    cost: float = 0.0


def composite_score(result: EvalResult, weights: ObjectiveWeights) -> float:
    """Compute a scalar score. Higher is always better.

    Latency and cost are subtracted (penalized).
    Cost is scaled to milli-dollars so it is comparable in magnitude to accuracy.
    """
    return (
        weights.accuracy * result.accuracy
        - weights.latency * result.avg_latency_s
        - weights.cost * (result.estimated_cost_usd * 1000)
    )
