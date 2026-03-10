"""ExperimentLoop — orchestrates the self-improvement cycle."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from autoagentlab.agent import Agent
from autoagentlab.evaluator import Evaluator, EvalResult
from autoagentlab.researcher import Researcher
from autoagentlab.mutator import Mutator


# ── ANSI helpers ────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def _bar(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return f"{'█' * filled}{'░' * (width - filled)}"


@dataclass
class IterationRecord:
    """Record of a single experiment iteration."""
    iteration: int
    accuracy: float
    prompt: str
    suggestion: str = ""
    failures: list[dict] = field(default_factory=list)
    accepted: bool = False


class ExperimentLoop:
    """Runs the full agent self-improvement loop.

    Flow per iteration:
      1. Evaluate current agent → accuracy + failures
      2. Researcher analyzes failures → suggestion
      3. Mutator rewrites prompt using suggestion → candidate agent
      4. Evaluate candidate → accept if better
      5. Repeat
    """

    def __init__(
        self,
        agent: Agent,
        tasks: list[list[str]],
        *,
        max_iterations: int = 5,
    ):
        self.agent = agent
        self.tasks = tasks
        self.max_iterations = max_iterations
        self.evaluator = Evaluator()
        self.researcher = Researcher()
        self.mutator = Mutator()
        self.history: list[IterationRecord] = []

    # ── Main entry point ────────────────────────────────────────────
    def run(self) -> list[IterationRecord]:
        """Execute the experiment loop and return the history."""
        self._print_header()

        for i in range(1, self.max_iterations + 1):
            record = self._run_iteration(i)
            self.history.append(record)

        self._print_summary()
        return self.history

    # ── Single iteration ────────────────────────────────────────────
    def _run_iteration(self, iteration: int) -> IterationRecord:
        self._print_iteration_start(iteration)

        # 1. Evaluate
        result = self.evaluator.evaluate(self.agent, self.tasks)
        self._print_eval(result)

        record = IterationRecord(
            iteration=iteration,
            accuracy=result.accuracy,
            prompt=self.agent.prompt,
            failures=result.failures,
        )

        # If perfect, stop early
        if result.accuracy == 1.0:
            self._print_perfect()
            record.accepted = True
            return record

        # 2. Research
        sys.stdout.write(f"  {DIM}🔬 Analyzing failures...{RESET}\n")
        sys.stdout.flush()
        suggestion = self.researcher.analyze(self.agent.prompt, result.failures)
        record.suggestion = suggestion
        self._print_suggestion(suggestion)

        # 3. Mutate
        sys.stdout.write(f"  {DIM}🧬 Generating improved prompt...{RESET}\n")
        sys.stdout.flush()
        new_prompt = self.mutator.improve(self.agent.prompt, suggestion)

        # 4. Evaluate candidate
        candidate = self.agent.clone(new_prompt)
        new_result = self.evaluator.evaluate(candidate, self.tasks)

        # 5. Accept or reject
        if new_result.accuracy >= result.accuracy:
            self.agent = candidate
            record.accepted = True
            self._print_accepted(new_result.accuracy)
        else:
            self._print_rejected(result.accuracy, new_result.accuracy)

        return record

    # ── Pretty printing ─────────────────────────────────────────────
    def _print_header(self) -> None:
        print()
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  🧪  AutoAgentLab — Agent Evolution{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"  {DIM}Tasks: {len(self.tasks)} | Max iterations: {self.max_iterations}{RESET}")
        print(f"  {DIM}Model: {self.agent.model}{RESET}")
        print()

    def _print_iteration_start(self, i: int) -> None:
        print(f"{BOLD}{YELLOW}── Iteration {i} ─────────────────────────────────────{RESET}")

    def _print_eval(self, result: EvalResult) -> None:
        color = GREEN if result.accuracy >= 0.8 else YELLOW if result.accuracy >= 0.5 else RED
        print(f"  📊 Accuracy: {color}{BOLD}{result.accuracy_pct}{RESET}  "
              f"{DIM}{_bar(result.accuracy)}{RESET}  "
              f"({result.correct}/{result.total})")
        if result.failures:
            print(f"  {DIM}❌ {len(result.failures)} failure(s){RESET}")

    def _print_suggestion(self, suggestion: str) -> None:
        short = suggestion[:120].replace("\n", " ")
        print(f"  {MAGENTA}💡 Suggestion:{RESET} {short}...")

    def _print_accepted(self, new_acc: float) -> None:
        print(f"  {GREEN}✅ Mutation accepted — new accuracy: {new_acc * 100:.0f}%{RESET}")
        print()

    def _print_rejected(self, old_acc: float, new_acc: float) -> None:
        print(f"  {RED}❌ Mutation rejected — "
              f"{new_acc * 100:.0f}% < {old_acc * 100:.0f}% (keeping current){RESET}")
        print()

    def _print_perfect(self) -> None:
        print(f"  {GREEN}{BOLD}🎉 Perfect score! No improvement needed.{RESET}")
        print()

    def _print_summary(self) -> None:
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  📈  Evolution Summary{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        for rec in self.history:
            marker = "✅" if rec.accepted else "❌"
            color = GREEN if rec.accuracy >= 0.8 else YELLOW if rec.accuracy >= 0.5 else RED
            print(f"  {marker} Iteration {rec.iteration}: "
                  f"{color}{BOLD}{rec.accuracy * 100:.0f}%{RESET}  "
                  f"{DIM}{_bar(rec.accuracy, 20)}{RESET}")

        best = max(self.history, key=lambda r: r.accuracy)
        print()
        print(f"  {BOLD}Best: Iteration {best.iteration} — {best.accuracy * 100:.0f}%{RESET}")
        print(f"  {DIM}Final prompt:{RESET}")
        for line in self.agent.prompt.split("\n")[:5]:
            print(f"    {DIM}{line}{RESET}")
        print()
