"""Tests for evaluator latency/token tracking — no LLM calls required."""

from autoagentlab.evaluator import Evaluator, EvalResult


class _FakeAgent:
    """Agent stub that returns preset answers without calling any LLM."""

    def __init__(self, answers: dict | None = None, tokens_per_call: int = 5):
        self.model = "gpt-4o-mini"
        self._answers = answers or {}
        self._tokens = tokens_per_call
        self._last_usage = 0

    def run(self, question: str) -> str:
        self._last_usage = self._tokens
        return self._answers.get(question, "default answer")


class _BrokenAgent:
    """Agent stub that always raises on run()."""

    model = "gpt-4o-mini"
    _last_usage = 0

    def run(self, question: str) -> str:
        raise RuntimeError("LLM unavailable")


# ── EvalResult properties ───────────────────────────────────────────

def test_eval_result_accuracy():
    r = EvalResult(total=4, correct=3)
    assert r.accuracy == 0.75


def test_eval_result_accuracy_pct():
    r = EvalResult(total=4, correct=3)
    assert r.accuracy_pct == "75%"


def test_eval_result_zero_total():
    r = EvalResult()
    assert r.accuracy == 0.0
    assert r.avg_latency_s == 0.0


def test_eval_result_avg_latency():
    r = EvalResult(total=4)
    r.total_latency_s = 8.0
    assert r.avg_latency_s == 2.0


def test_eval_result_estimated_cost():
    r = EvalResult()
    r.total_tokens = 1_000_000
    assert r.estimated_cost_usd > 0.0


# ── Evaluator.evaluate() ────────────────────────────────────────────

def test_all_correct():
    tasks = [["Q1", "A"], ["Q2", "B"]]
    agent = _FakeAgent({"Q1": "A", "Q2": "B"})
    result = Evaluator().evaluate(agent, tasks)
    assert result.accuracy == 1.0
    assert result.failures == []


def test_partial_correct():
    tasks = [["Q1", "yes"], ["Q2", "no"]]
    agent = _FakeAgent({"Q1": "yes"})   # Q2 returns "default answer", not "no"
    result = Evaluator().evaluate(agent, tasks)
    assert result.accuracy == 0.5
    assert len(result.failures) == 1


def test_tokens_accumulated():
    tasks = [["Q1", "A"], ["Q2", "B"], ["Q3", "C"]]
    agent = _FakeAgent(tokens_per_call=7)
    result = Evaluator().evaluate(agent, tasks)
    assert result.total_tokens == 21   # 3 tasks × 7 tokens


def test_latency_accumulated():
    tasks = [["Q1", "A"], ["Q2", "B"]]
    agent = _FakeAgent()
    result = Evaluator().evaluate(agent, tasks)
    assert result.total_latency_s >= 0.0
    assert result.avg_latency_s >= 0.0


def test_error_recorded_not_raised():
    tasks = [["Q?", "A"]]
    result = Evaluator().evaluate(_BrokenAgent(), tasks)
    assert result.accuracy == 0.0
    assert len(result.failures) == 1
    assert "[ERROR]" in result.failures[0]["response"]


def test_case_insensitive_answer_check():
    tasks = [["Q", "Paris"]]
    agent = _FakeAgent({"Q": "The capital is paris."})
    result = Evaluator().evaluate(agent, tasks)
    assert result.accuracy == 1.0
