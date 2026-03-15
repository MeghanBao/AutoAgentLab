"""ExperimentLoop — orchestrates the self-improvement cycle."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from autoagentlab.agent import Agent
from autoagentlab.evaluator import Evaluator, EvalResult
from autoagentlab.researcher import Researcher
from autoagentlab.mutator import Mutator
from autoagentlab.scorer import ObjectiveWeights, composite_score
from autoagentlab._display import BOLD, DIM, CYAN, GREEN, YELLOW, RED, MAGENTA, RESET, bar


@dataclass
class IterationRecord:
    """Record of a single experiment iteration."""

    iteration: int
    accuracy: float           # accuracy of current agent before mutation
    prompt: str               # current agent's prompt before mutation
    agent_id: str = ""        # lineage: ID of the agent evaluated this iteration
    parent_id: str | None = None   # lineage: parent agent's ID
    generation: int = 0       # generation number of the evaluated agent
    suggestion: str = ""
    failures: list[dict] = field(default_factory=list)
    accepted: bool = False
    candidate_accuracy: float | None = None  # accuracy of the proposed mutation


class ExperimentLoop:
    """Runs the full agent self-improvement loop.

    Flow per iteration:
      1. Evaluate current agent → accuracy + failures
      2. Researcher analyzes failures → improvement suggestion
      3. Mutator rewrites prompt using suggestion → candidate agent
      4. Evaluate candidate → accept if composite score improves
      5. Repeat

    Multi-objective:
        Pass *weights* to penalize cost or latency alongside accuracy.
        Example: ObjectiveWeights(accuracy=1.0, latency=0.5, cost=0.2)

    Agent Lineage:
        After the run, *_print_summary* shows a visual lineage tree
        tracking which mutations were accepted or rejected.
    """

    def __init__(
        self,
        agent: Agent,
        tasks: list[list[str]],
        *,
        max_iterations: int = 5,
        weights: ObjectiveWeights | None = None,
    ):
        self.agent = agent
        self.tasks = tasks
        self.max_iterations = max_iterations
        self.weights = weights or ObjectiveWeights()
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
            agent_id=self.agent.agent_id,
            parent_id=self.agent.parent_id,
            generation=self.agent.generation,
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
        record.candidate_accuracy = new_result.accuracy

        # 5. Accept or reject (based on composite score)
        current_score = composite_score(result, self.weights)
        new_score = composite_score(new_result, self.weights)

        if new_score >= current_score:
            self.agent = candidate
            record.accepted = True
            self._print_accepted(new_result)
        else:
            self._print_rejected(result, new_result)

        return record

    # ── Pretty printing ─────────────────────────────────────────────
    def _print_header(self) -> None:
        print()
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  🧪  AutoAgentLab — Agent Evolution{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"  {DIM}Tasks: {len(self.tasks)} | Max iterations: {self.max_iterations}{RESET}")
        print(f"  {DIM}Model: {self.agent.model}{RESET}")
        if self.weights.latency > 0 or self.weights.cost > 0:
            print(
                f"  {DIM}Objectives: accuracy={self.weights.accuracy:.1f}  "
                f"latency={self.weights.latency:.1f}  "
                f"cost={self.weights.cost:.1f}{RESET}"
            )
        print()

    def _print_iteration_start(self, i: int) -> None:
        print(f"{BOLD}{YELLOW}── Iteration {i} ─────────────────────────────────────{RESET}")

    def _print_eval(self, result: EvalResult) -> None:
        color = GREEN if result.accuracy >= 0.8 else YELLOW if result.accuracy >= 0.5 else RED
        print(
            f"  📊 Accuracy: {color}{BOLD}{result.accuracy_pct}{RESET}  "
            f"{DIM}{bar(result.accuracy)}{RESET}  "
            f"({result.correct}/{result.total})"
        )
        if result.total_latency_s > 0:
            print(
                f"  {DIM}⏱  {result.avg_latency_s:.1f}s/task  "
                f"💰 ${result.estimated_cost_usd:.5f} total{RESET}"
            )
        if result.failures:
            print(f"  {DIM}❌ {len(result.failures)} failure(s){RESET}")

    def _print_suggestion(self, suggestion: str) -> None:
        short = suggestion[:120].replace("\n", " ")
        print(f"  {MAGENTA}💡 Suggestion:{RESET} {short}...")

    def _print_accepted(self, new_result: EvalResult) -> None:
        print(
            f"  {GREEN}✅ Mutation accepted — "
            f"new accuracy: {new_result.accuracy * 100:.0f}%{RESET}"
        )
        print()

    def _print_rejected(self, old_result: EvalResult, new_result: EvalResult) -> None:
        print(
            f"  {RED}❌ Mutation rejected — "
            f"{new_result.accuracy * 100:.0f}% < {old_result.accuracy * 100:.0f}%"
            f" (keeping current){RESET}"
        )
        print()

    def _print_perfect(self) -> None:
        print(f"  {GREEN}{BOLD}🎉 Perfect score! No improvement needed.{RESET}")
        print()

    def _print_lineage_tree(self) -> None:
        """Visual lineage: each row shows one iteration's evaluation and mutation outcome."""
        if not self.history:
            return
        print(f"  {BOLD}📜 Prompt Lineage{RESET}")
        for rec in self.history:
            acc_color = GREEN if rec.accuracy >= 0.8 else YELLOW if rec.accuracy >= 0.5 else RED
            line = (
                f"    Gen {rec.generation}  "
                f"{acc_color}{rec.accuracy * 100:.0f}%{RESET}  "
                f"[{DIM}id={rec.agent_id}{RESET}]  →  "
            )
            if rec.candidate_accuracy is not None:
                cand_color = (
                    GREEN if rec.candidate_accuracy >= 0.8
                    else YELLOW if rec.candidate_accuracy >= 0.5
                    else RED
                )
                marker = "✅" if rec.accepted else "❌"
                line += (
                    f"candidate {cand_color}{rec.candidate_accuracy * 100:.0f}%{RESET} {marker}"
                )
            else:
                line += f"{GREEN}perfect ✨{RESET}"
            print(line)
        print()

    def _print_summary(self) -> None:
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  📈  Evolution Summary{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        for rec in self.history:
            marker = "✅" if rec.accepted else "❌"
            color = GREEN if rec.accuracy >= 0.8 else YELLOW if rec.accuracy >= 0.5 else RED
            print(
                f"  {marker} Iteration {rec.iteration}: "
                f"{color}{BOLD}{rec.accuracy * 100:.0f}%{RESET}  "
                f"{DIM}{bar(rec.accuracy, 20)}{RESET}"
            )

        best = max(self.history, key=lambda r: r.accuracy)
        print()
        print(f"  {BOLD}Best: Iteration {best.iteration} — {best.accuracy * 100:.0f}%{RESET}")
        self._print_lineage_tree()
        print(f"  {DIM}Final prompt:{RESET}")
        for line in self.agent.prompt.split("\n")[:5]:
            print(f"    {DIM}{line}{RESET}")
        print()
