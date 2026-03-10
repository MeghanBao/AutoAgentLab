"""Researcher module — analyzes agent failures and suggests improvements."""

from __future__ import annotations

from autoagentlab.agent import llm


class Researcher:
    """Analyzes failure cases and produces improvement suggestions."""

    ANALYSIS_PROMPT = """You are an AI research scientist analyzing an AI agent's failures.

The agent uses this system prompt:
---
{agent_prompt}
---

The agent answered the following questions INCORRECTLY:

{failures}

Analyze the failure patterns and provide a concise, actionable suggestion
for how to improve the agent's system prompt so it answers more accurately.

Focus on:
1. What patterns of errors do you see?
2. What reasoning strategies would help?
3. What specific instructions should be added to the prompt?

Respond with ONLY the improvement suggestion, no preamble."""

    def analyze(self, agent_prompt: str, failures: list[dict]) -> str:
        """Analyze failures and return an improvement suggestion.

        Args:
            agent_prompt: The current agent system prompt.
            failures: List of dicts with 'question', 'expected', 'response'.

        Returns:
            A string suggestion for prompt improvement.
        """
        if not failures:
            return "No failures to analyze. The agent is performing perfectly."

        failure_text = "\n".join(
            f"Q: {f['question']}\n"
            f"Expected: {f['expected']}\n"
            f"Agent answered: {f['response']}\n"
            for f in failures
        )

        prompt = self.ANALYSIS_PROMPT.format(
            agent_prompt=agent_prompt,
            failures=failure_text,
        )

        return llm(prompt)
