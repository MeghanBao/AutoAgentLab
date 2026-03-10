"""Mutator module — rewrites the agent's prompt based on improvement suggestions."""

from __future__ import annotations

from autoagentlab.agent import llm


class Mutator:
    """Rewrites an agent's system prompt using a research suggestion."""

    MUTATION_PROMPT = """You are an AI prompt engineer. Your job is to improve an AI agent's system prompt.

Current system prompt:
---
{current_prompt}
---

Improvement suggestion from the research team:
---
{suggestion}
---

Rewrite the system prompt to incorporate the suggestion. The new prompt should:
1. Keep the original intent and role
2. Add the suggested improvements
3. Be clear and concise
4. Not exceed 500 words

Respond with ONLY the new system prompt, no preamble or explanation."""

    def improve(self, current_prompt: str, suggestion: str) -> str:
        """Generate an improved prompt by applying the suggestion.

        Args:
            current_prompt: The agent's current system prompt.
            suggestion: The researcher's improvement suggestion.

        Returns:
            A new, improved system prompt string.
        """
        prompt = self.MUTATION_PROMPT.format(
            current_prompt=current_prompt,
            suggestion=suggestion,
        )

        return llm(prompt).strip()
