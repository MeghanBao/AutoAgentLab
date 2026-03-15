"""Agent module — an LLM-powered agent with a mutable system prompt."""

from __future__ import annotations

import os
import re
import uuid

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


# ── Tool-call protocol ──────────────────────────────────────────────
# The LLM signals a tool call by writing on its own line:
#   TOOL: tool_name
#   ARGS: arguments here

_TOOL_CALL_RE = re.compile(
    r"^TOOL:\s*(\w+)\s*\nARGS:\s*(.+?)(?:\n|$)",
    re.MULTILINE,
)
_MAX_TOOL_STEPS = 5


# ── Agent ───────────────────────────────────────────────────────────

class Agent:
    """An AI agent defined by a system prompt.

    The agent answers questions by combining its system prompt with the
    user's question and sending them to an LLM.

    Tool Discovery:
        When *tools_enabled=True*, the agent runs a ReAct-style loop:
        it detects TOOL/ARGS lines in the LLM response, executes the
        requested tool, and feeds the result back before answering.

    Agent Lineage:
        Every agent has a unique *agent_id*. When cloned, the child
        records the parent's *agent_id* in *parent_id*, enabling full
        lineage tracking across generations.

    Usage Tracking:
        After each *run()* call, *_last_usage* holds the total tokens
        consumed (used by Evaluator to accumulate cost/latency data).
    """

    def __init__(
        self,
        prompt: str,
        *,
        model: str | None = None,
        tools_enabled: bool = False,
    ):
        self.prompt = prompt
        self.model = model or os.getenv("AUTOAGENTLAB_MODEL", "gpt-4o-mini")
        self.tools_enabled = tools_enabled
        self.generation: int = 0

        # Lineage tracking
        self.agent_id: str = uuid.uuid4().hex[:8]
        self.parent_id: str | None = None

        # Token usage from the most recent run() call
        self._last_usage: int = 0

    def run(self, question: str) -> str:
        """Run the agent on a single question and return the answer string."""
        from autoagentlab.tools import TOOL_DESCRIPTIONS, run_tool  # noqa: PLC0415

        system = self.prompt
        if self.tools_enabled:
            system = system + TOOL_DESCRIPTIONS

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        client = _get_client()
        total_tokens = 0

        for _ in range(_MAX_TOOL_STEPS):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            text = response.choices[0].message.content or ""
            if response.usage:
                total_tokens += response.usage.total_tokens

            # No tools or no tool call in response → return immediately
            if not self.tools_enabled:
                self._last_usage = total_tokens
                return text

            match = _TOOL_CALL_RE.search(text)
            if match is None:
                self._last_usage = total_tokens
                return text

            # Execute tool and feed result back into the conversation
            tool_name = match.group(1).strip()
            tool_args = match.group(2).strip()
            result = run_tool(tool_name, tool_args)

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"TOOL RESULT: {result}"})

        self._last_usage = total_tokens
        return text

    def clone(self, new_prompt: str) -> "Agent":
        """Create a next-generation agent with a new prompt."""
        child = Agent(new_prompt, model=self.model, tools_enabled=self.tools_enabled)
        child.generation = self.generation + 1
        child.parent_id = self.agent_id
        return child

    def __repr__(self) -> str:
        short = self.prompt[:60].replace("\n", " ")
        return f"Agent(gen={self.generation}, id={self.agent_id}, prompt='{short}...')"
