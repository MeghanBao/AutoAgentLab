"""Population-based evolution — run N agents in parallel per iteration."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from autoagentlab.agent import Agent
from autoagentlab.evaluator import Evaluator, EvalResult
from autoagentlab.researcher import Researcher
from autoagentlab.mutator import Mutator
from autoagentlab.scorer import ObjectiveWeights, composite_score
from autoagentlab._display import BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET, bar


@dataclass
class PopulationRecord:
    """Result of one agent for one iteration."""

    iteration: int
    rank: int         # 1 = best within this iteration
    agent_id: str
    generation: int
    accuracy: float
    latency_s: float
    cost_usd: float
    is_elite: bool


class PopulationLoop:
    """Evolve a population of N agents in parallel.

    Each iteration:
      1. Evaluate all agents in parallel (ThreadPoolExecutor).
      2. Sort by composite score (descending).
      3. Keep the top *elite_size* agents unchanged (elitism).
      4. Fill remaining slots by mutating elite agents (researcher → mutator).

    The final population's best agent is available via *best_agent()*.
    """

    def __init__(
        self,
        agent: Agent,
        tasks: list[list[str]],
        *,
        population_size: int = 4,
        max_iterations: int = 5,
        elite_size: int | None = None,
        weights: ObjectiveWeights | None = None,
        max_workers: int | None = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be >= 2")

        self.tasks = tasks
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.elite_size = elite_size if elite_size is not None else max(1, population_size // 2)
        self.weights = weights or ObjectiveWeights()
        self.max_workers = max_workers or population_size
        self.evaluator = Evaluator()
        self.researcher = Researcher()
        self.mutator = Mutator()
        self.history: list[list[PopulationRecord]] = []

        # Seed: index-0 keeps the original agent; rest are clones with fresh IDs
        self.population: list[Agent] = [
            agent if i == 0 else agent.clone(agent.prompt)
            for i in range(population_size)
        ]

    def run(self) -> list[list[PopulationRecord]]:
        """Execute the population loop and return per-iteration records."""
        self._print_header()

        for i in range(1, self.max_iterations + 1):
            records = self._run_iteration(i)
            self.history.append(records)

            # Early stop if the champion is perfect
            if records[0].accuracy >= 1.0:
                print(f"  {GREEN}{BOLD}🎉 Champion reached 100%! Stopping early.{RESET}\n")
                break

        self._print_summary()
        return self.history

    def best_agent(self) -> Agent:
        """Return the current best agent (rank-1 after the last iteration)."""
        return self.population[0]

    def _run_iteration(self, iteration: int) -> list[PopulationRecord]:
        print(
            f"{BOLD}{YELLOW}── Iteration {iteration} "
            f"(pop={self.population_size}) {'─' * 28}{RESET}"
        )
        sys.stdout.flush()

        # 1. Parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self.evaluator.evaluate, ag, self.tasks): idx
                for idx, ag in enumerate(self.population)
            }
            idx_results: dict[int, EvalResult] = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    idx_results[idx] = future.result()
                except Exception as exc:
                    sys.stderr.write(f"  [warn] Agent {idx} evaluation error: {exc}\n")
                    idx_results[idx] = EvalResult(total=len(self.tasks))

        # 2. Sort by composite score (descending)
        order = sorted(
            range(self.population_size),
            key=lambda i: composite_score(idx_results[i], self.weights),
            reverse=True,
        )
        self.population = [self.population[i] for i in order]
        sorted_results = [idx_results[i] for i in order]

        # 3. Print table
        records: list[PopulationRecord] = []
        for rank, (agent, result) in enumerate(zip(self.population, sorted_results), 1):
            is_elite = rank <= self.elite_size
            crown = "👑" if is_elite else "  "
            color = GREEN if result.accuracy >= 0.8 else YELLOW if result.accuracy >= 0.5 else RED
            print(
                f"  {crown} #{rank} {color}{BOLD}{result.accuracy_pct}{RESET}  "
                f"{DIM}{bar(result.accuracy, 20)}{RESET}  "
                f"gen={agent.generation} id={agent.agent_id}  "
                f"⏱ {result.avg_latency_s:.1f}s  💰 ${result.estimated_cost_usd:.5f}"
            )
            records.append(PopulationRecord(
                iteration=iteration,
                rank=rank,
                agent_id=agent.agent_id,
                generation=agent.generation,
                accuracy=result.accuracy,
                latency_s=result.avg_latency_s,
                cost_usd=result.estimated_cost_usd,
                is_elite=is_elite,
            ))

        # 4. Elitism + mutation
        elites = self.population[:self.elite_size]
        elite_results = sorted_results[:self.elite_size]
        slots_needed = self.population_size - self.elite_size

        if slots_needed > 0:
            sys.stdout.write(
                f"  {DIM}🧬 Generating {slots_needed} mutation(s)...{RESET}\n"
            )
            sys.stdout.flush()
            new_population = list(elites)
            for i in range(slots_needed):
                parent = elites[i % len(elites)]
                parent_result = elite_results[i % len(elite_results)]
                suggestion = self.researcher.analyze(parent.prompt, parent_result.failures)
                new_prompt = self.mutator.improve(parent.prompt, suggestion)
                new_population.append(parent.clone(new_prompt))
            self.population = new_population

        print()
        return records

    def _print_header(self) -> None:
        print()
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  🧬  AutoAgentLab — Population Evolution{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(
            f"  {DIM}Tasks: {len(self.tasks)} | "
            f"Population: {self.population_size} | "
            f"Elite: {self.elite_size} | "
            f"Max iters: {self.max_iterations}{RESET}"
        )
        print(f"  {DIM}Model: {self.population[0].model}{RESET}")
        if self.weights.latency > 0 or self.weights.cost > 0:
            print(
                f"  {DIM}Objectives: accuracy={self.weights.accuracy:.1f}  "
                f"latency={self.weights.latency:.1f}  "
                f"cost={self.weights.cost:.1f}{RESET}"
            )
        print()

    def _print_summary(self) -> None:
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  📈  Population Summary{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")

        for records in self.history:
            best = records[0]
            color = GREEN if best.accuracy >= 0.8 else YELLOW if best.accuracy >= 0.5 else RED
            print(
                f"  Iteration {best.iteration}: "
                f"best={color}{BOLD}{best.accuracy * 100:.0f}%{RESET}  "
                f"{DIM}{bar(best.accuracy, 20)}{RESET}"
            )

        if self.history:
            all_records = [r for recs in self.history for r in recs]
            overall_best = max(all_records, key=lambda r: r.accuracy)
            print()
            print(
                f"  {BOLD}Best ever: Iteration {overall_best.iteration}, "
                f"#{overall_best.rank} — {overall_best.accuracy * 100:.0f}%{RESET}"
            )
            print(f"\n  {DIM}Champion's prompt (first 5 lines):{RESET}")
            for line in self.population[0].prompt.split("\n")[:5]:
                print(f"    {DIM}{line}{RESET}")
        print()
