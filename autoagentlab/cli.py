"""CLI entry point for AutoAgentLab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoagentlab.agent import Agent
from autoagentlab.loop import ExperimentLoop
from autoagentlab.scorer import ObjectiveWeights


DEFAULT_PROMPT = "You are a helpful assistant. Answer questions accurately and concisely."

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def _load_benchmark(name: str) -> list[list[str]]:
    """Load a benchmark JSON file by name."""
    path = BENCHMARKS_DIR / f"{name}.json"
    if not path.exists():
        print(f"Error: benchmark '{name}' not found at {path}")
        available = ", ".join(p.stem for p in sorted(BENCHMARKS_DIR.glob("*.json")))
        print(f"Available benchmarks: {available}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment loop on the given benchmark."""
    tasks = _load_benchmark(args.benchmark)

    prompt = args.prompt or DEFAULT_PROMPT
    agent = Agent(prompt, model=args.model, tools_enabled=args.tools)

    weights = ObjectiveWeights(
        accuracy=1.0,
        latency=args.weight_latency,
        cost=args.weight_cost,
    )

    loop = ExperimentLoop(
        agent,
        tasks,
        max_iterations=args.iterations,
        weights=weights,
    )
    loop.run()


def _cmd_run_population(args: argparse.Namespace) -> None:
    """Run a population-based evolution loop on the given benchmark."""
    from autoagentlab.population import PopulationLoop  # noqa: PLC0415

    tasks = _load_benchmark(args.benchmark)
    prompt = args.prompt or DEFAULT_PROMPT
    agent = Agent(prompt, model=args.model, tools_enabled=args.tools)

    weights = ObjectiveWeights(
        accuracy=1.0,
        latency=args.weight_latency,
        cost=args.weight_cost,
    )

    loop = PopulationLoop(
        agent,
        tasks,
        population_size=args.population,
        max_iterations=args.iterations,
        elite_size=args.elite,
        weights=weights,
        max_workers=args.workers,
    )
    loop.run()


def _cmd_list(args: argparse.Namespace) -> None:
    """List available benchmarks."""
    print("Available benchmarks:\n")
    for path in sorted(BENCHMARKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {path.stem:<20} ({len(data)} tasks)")
    print()


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared between 'run' and 'run-population'."""
    parser.add_argument("benchmark", help="Benchmark name (e.g. 'qa')")
    parser.add_argument(
        "--iterations", "-n",
        type=int, default=5,
        help="Number of improvement iterations (default: 5)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Initial agent system prompt",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        default=False,
        help="Enable built-in tools (calculator, python) for the agent",
    )
    parser.add_argument(
        "--weight-latency",
        type=float, default=0.0, metavar="W",
        help="Penalty weight for avg latency (seconds) in accept/reject decisions (default: 0)",
    )
    parser.add_argument(
        "--weight-cost",
        type=float, default=0.0, metavar="W",
        help="Penalty weight for cost (milli-dollars) in accept/reject decisions (default: 0)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autoagentlab",
        description="AutoAgentLab — Automated experimentation for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single-agent improvement loop",
    )
    _add_common_run_args(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    # ── run-population ───────────────────────────────────────────────
    pop_parser = subparsers.add_parser(
        "run-population",
        help="Run a population-based parallel evolution loop",
    )
    _add_common_run_args(pop_parser)
    pop_parser.add_argument(
        "--population",
        type=int, default=4, metavar="N",
        help="Number of agents in the population (default: 4)",
    )
    pop_parser.add_argument(
        "--elite",
        type=int, default=None, metavar="K",
        help="Number of elite agents kept each iteration (default: population // 2)",
    )
    pop_parser.add_argument(
        "--workers",
        type=int, default=None, metavar="W",
        help="Max parallel evaluation workers (default: population size)",
    )
    pop_parser.set_defaults(func=_cmd_run_population)

    # ── list ─────────────────────────────────────────────────────────
    list_parser = subparsers.add_parser("list", help="List available benchmarks")
    list_parser.set_defaults(func=_cmd_list)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
