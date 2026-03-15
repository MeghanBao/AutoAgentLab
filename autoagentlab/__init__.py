"""AutoAgentLab — Automated experimentation for AI agents."""

from autoagentlab.agent import Agent
from autoagentlab.evaluator import Evaluator
from autoagentlab.researcher import Researcher
from autoagentlab.mutator import Mutator
from autoagentlab.loop import ExperimentLoop
from autoagentlab.population import PopulationLoop
from autoagentlab.scorer import ObjectiveWeights, composite_score
from autoagentlab.tools import TOOLS, run_tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Evaluator",
    "Researcher",
    "Mutator",
    "ExperimentLoop",
    "PopulationLoop",
    "ObjectiveWeights",
    "composite_score",
    "TOOLS",
    "run_tool",
]
