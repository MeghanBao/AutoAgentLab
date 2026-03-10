"""Agent module — an LLM-powered agent with a mutable system prompt."""

from __future__ import annotations

import os
from openai import OpenAI


def _get_client() -> OpenAI:
    """Create an OpenAI client (supports any OpenAI-compatible provider)."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def llm(prompt: str, *, model: str | None = None) -> str:
    """Call an LLM with a single user message and return the response text."""
    model = model or os.getenv("AUTOAGENTLAB_MODEL", "gpt-4o-mini")
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


class Agent:
    """An AI agent defined by a system prompt.

    The agent answers questions by combining its system prompt with the
    user's question and sending them to an LLM.
    """

    def __init__(self, prompt: str, *, model: str | None = None):
        self.prompt = prompt
        self.model = model or os.getenv("AUTOAGENTLAB_MODEL", "gpt-4o-mini")
        self.generation: int = 0

    def run(self, question: str) -> str:
        """Run the agent on a single question."""
        full_prompt = f"{self.prompt}\n\nQuestion: {question}\nAnswer:"
        return llm(full_prompt, model=self.model)

    def clone(self, new_prompt: str) -> "Agent":
        """Create a next-generation agent with a new prompt."""
        child = Agent(new_prompt, model=self.model)
        child.generation = self.generation + 1
        return child

    def __repr__(self) -> str:
        short = self.prompt[:60].replace("\n", " ")
        return f"Agent(gen={self.generation}, prompt='{short}...')"
