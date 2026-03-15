"""Tests for built-in tools — no LLM calls required."""

from autoagentlab.tools import run_tool, TOOL_DESCRIPTIONS, TOOLS


# ── calculator ──────────────────────────────────────────────────────

def test_calculator_addition():
    assert run_tool("calculator", "2 + 2") == "4"


def test_calculator_multiplication():
    assert run_tool("calculator", "15 * 13") == "195"


def test_calculator_power():
    assert run_tool("calculator", "2 ** 10") == "1024"


def test_calculator_caret_power():
    assert run_tool("calculator", "2 ^ 10") == "1024"


def test_calculator_sqrt():
    assert run_tool("calculator", "sqrt(16)") == "4"


def test_calculator_float_division():
    result = run_tool("calculator", "10 / 4")
    assert result == "2.5"


def test_calculator_modulo():
    assert run_tool("calculator", "17 % 5") == "2"


def test_calculator_floor_division():
    assert run_tool("calculator", "17 // 5") == "3"


def test_calculator_nested():
    assert run_tool("calculator", "sqrt(144) + 2 ** 3") == "20"


def test_calculator_invalid_syntax():
    result = run_tool("calculator", "not valid !!")
    assert "Error" in result or "ToolError" in result


# ── python ──────────────────────────────────────────────────────────

def test_python_print():
    result = run_tool("python", "print('hello')")
    assert result == "hello"


def test_python_arithmetic():
    result = run_tool("python", "print(sum(range(10)))")
    assert result == "45"


def test_python_multiline():
    code = "x = 3\ny = 4\nprint(x * y)"
    assert run_tool("python", code) == "12"


def test_python_no_output():
    result = run_tool("python", "x = 1 + 1")
    assert result == "(no output)"


def test_python_blocked_import_os():
    result = run_tool("python", "import os")
    assert "[ToolError]" in result


def test_python_blocked_import_sys():
    result = run_tool("python", "import sys")
    assert "[ToolError]" in result


def test_python_runtime_error():
    result = run_tool("python", "print(1 / 0)")
    assert "[ToolError]" in result


# ── registry ────────────────────────────────────────────────────────

def test_unknown_tool():
    result = run_tool("nonexistent", "args")
    assert "[ToolError]" in result
    assert "nonexistent" in result


def test_tool_descriptions_is_string():
    assert isinstance(TOOL_DESCRIPTIONS, str)
    assert "calculator" in TOOL_DESCRIPTIONS
    assert "TOOL:" in TOOL_DESCRIPTIONS
    assert "ARGS:" in TOOL_DESCRIPTIONS


def test_tools_registry_keys():
    assert "calculator" in TOOLS
    assert "python" in TOOLS


def test_tool_callable():
    tool = TOOLS["calculator"]
    assert tool("1 + 1") == "2"
