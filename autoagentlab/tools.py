"""Built-in tools available to agents — calculator and Python sandbox."""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    """A named, callable tool with a description."""

    name: str
    description: str
    _fn: Callable[[str], str] = field(repr=False)

    def __call__(self, args: str) -> str:
        try:
            return self._fn(args.strip())
        except Exception as exc:
            return f"[ToolError] {exc}"


# ── Safe math evaluator ─────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return float(_SAFE_NAMES[node.id])  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_NAMES.get(node.func.id)
        if fn and callable(fn):
            args_vals = [_safe_eval(a) for a in node.args]
            return float(fn(*args_vals))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _calculator(expression: str) -> str:
    expr = expression.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    result = _safe_eval(tree.body)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return f"{result:.6g}"


# ── Python sandbox ──────────────────────────────────────────────────

_BLOCKED_RE = re.compile(
    r"\b(import\s+os|import\s+sys|import\s+subprocess|__import__|open\s*\()\b"
)

_SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in (
        "print", "len", "range", "int", "float", "str", "bool",
        "list", "dict", "set", "tuple", "sum", "min", "max",
        "abs", "round", "sorted", "enumerate", "zip", "map",
        "filter", "any", "all", "repr", "type", "isinstance",
    )
}


def _python(code: str) -> str:
    if _BLOCKED_RE.search(code):
        return "[ToolError] blocked: restricted operation detected"
    buf = io.StringIO()
    globs: dict = {"__builtins__": _SAFE_BUILTINS}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globs)  # noqa: S102
        return buf.getvalue().strip() or "(no output)"
    except Exception as exc:
        return f"[ToolError] {exc}"


# ── Public registry ─────────────────────────────────────────────────

TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description=(
            "Evaluate a mathematical expression. "
            "Supports +,-,*,/,**,% and math functions (sqrt, log, sin, cos, etc.)."
        ),
        _fn=_calculator,
    ),
    "python": Tool(
        name="python",
        description="Run Python code in a safe sandbox. Use print() to show results.",
        _fn=_python,
    ),
}

TOOL_DESCRIPTIONS: str = (
    "\n\n## Tools\n"
    "You have access to tools. To call a tool write on its own line:\n"
    "TOOL: tool_name\n"
    "ARGS: your arguments here\n\n"
    "The result will appear as 'TOOL RESULT: ...' which you should use in your answer.\n\n"
    "Available tools:\n"
    + "\n".join(f"  - {t.name}: {t.description}" for t in TOOLS.values())
)


def run_tool(name: str, args: str) -> str:
    """Dispatch a tool call by name. Returns an error string for unknown tools."""
    tool = TOOLS.get(name)
    if tool is None:
        available = ", ".join(TOOLS)
        return f"[ToolError] unknown tool: {name!r}. Available: {available}"
    return tool(args)
