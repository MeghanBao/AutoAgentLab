"""CLI entry point for AutoAgentLab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoagentlab.agent import Agent
from autoagentlab.loop import ExperimentLoop


DEFAULT_PROMPT = "You are a helpful assistant. Answer questions accurately and concisely."

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def _load_benchmark(name: str) -> list[list[str]]:
    """Load a benchmark JSON file by name."""
    path = BENCHMARKS_DIR / f"{name}.json"
    if not path.exists():
        print(f"Error: benchmark '{name}' not found at {path}")
        print(f"Available benchmarks: {', '.join(p.stem for p in BENCHMARKS_DIR.glob('*.json'))}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment loop on the given benchmark."""
    tasks = _load_benchmark(args.benchmark)

    prompt = args.prompt or DEFAULT_PROMPT
    agent = Agent(prompt, model=args.model)

    loop = ExperimentLoop(
        agent,
        tasks,
        max_iterations=args.iterations,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autoagentlab",
        description="AutoAgentLab — Automated experimentation for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run an experiment loop")
    run_parser.add_argument("benchmark", help="Name of the benchmark to run (e.g. 'qa')")
    run_parser.add_argument(
        "--iterations", "-n",
        type=int, default=5,
        help="Number of improvement iterations (default: 5)",
    )
    run_parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model to use (default: gpt-4o-mini)",
    )
    run_parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Initial agent system prompt",
    )
    run_parser.set_defaults(func=_cmd_run)

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
