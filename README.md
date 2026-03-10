<div align="center">

# 🧪 AutoAgentLab

**Automated experimentation for AI agents.**

*Agents run tasks, analyze failures, improve themselves, and repeat.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 💡 What is this?

AutoAgentLab turns **agent improvement** into an **automated research loop**. Instead of manually tweaking prompts, let your AI agents experiment on themselves:

```
Tasks → Agent → Evaluation → Research → Mutation → Repeat
```

> **Agents that research themselves.**

---

## ⚡ Quick Start

```bash
pip install -e .

export OPENAI_API_KEY="sk-..."

autoagentlab run qa
```

**30 seconds to your first experiment.**

---

## 🎬 Demo

```
════════════════════════════════════════════════════════════
  🧪  AutoAgentLab — Agent Evolution
════════════════════════════════════════════════════════════
  Tasks: 20 | Max iterations: 5
  Model: gpt-4o-mini

── Iteration 1 ─────────────────────────────────────
  📊 Accuracy: 54%  ██████████████████░░░░░░░░░░░░░░  (11/20)
  ❌ 9 failure(s)
  🔬 Analyzing failures...
  💡 Suggestion: Add step-by-step reasoning instructions...
  🧬 Generating improved prompt...
  ✅ Mutation accepted — new accuracy: 67%

── Iteration 2 ─────────────────────────────────────
  📊 Accuracy: 67%  ████████████████████████░░░░░░░░  (13/20)
  ❌ 7 failure(s)
  🔬 Analyzing failures...
  💡 Suggestion: Use structured output format...
  🧬 Generating improved prompt...
  ✅ Mutation accepted — new accuracy: 82%

── Iteration 3 ─────────────────────────────────────
  📊 Accuracy: 82%  █████████████████████████████░░░  (16/20)
  ...

════════════════════════════════════════════════════════════
  📈  Evolution Summary
════════════════════════════════════════════════════════════
  ✅ Iteration 1: 54%
  ✅ Iteration 2: 67%
  ✅ Iteration 3: 82%

  Best: Iteration 3 — 82%
```

> AI is doing research on itself. 🤯

---

## 🧠 How It Works

```
         ┌─────────────┐
         │   Benchmark  │   A set of (question, answer) tasks
         │    Tasks     │
         └──────┬───────┘
                │
                ▼
         ┌─────────────┐
         │    Agent     │   LLM + system prompt
         │              │
         └──────┬───────┘
                │ answers
                ▼
         ┌─────────────┐
         │  Evaluator   │   Score accuracy, collect failures
         │              │
         └──────┬───────┘
                │ failures
                ▼
         ┌─────────────┐
         │  Researcher  │   Analyze failure patterns
         │              │   → improvement suggestion
         └──────┬───────┘
                │ suggestion
                ▼
         ┌─────────────┐
         │   Mutator    │   Rewrite agent prompt
         │              │   using suggestion
         └──────┬───────┘
                │ new prompt
                ▼
         ┌─────────────┐
         │  New Agent   │   Accept if better,
         │              │   reject otherwise
         └──────┬───────┘
                │
                └──────── Repeat ─────────┘
```

---

## 📦 Project Structure

```
AutoAgentLab/
├── autoagentlab/
│   ├── __init__.py         # Public API
│   ├── agent.py            # LLM-powered agent
│   ├── evaluator.py        # Accuracy scoring
│   ├── researcher.py       # Failure analysis
│   ├── mutator.py          # Prompt rewriting
│   ├── loop.py             # Experiment orchestrator
│   └── cli.py              # CLI entry point
├── benchmarks/
│   └── qa.json             # 20 general-knowledge Q&A tasks
├── examples/
│   └── custom_benchmark.py # Programmatic usage example
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🔧 Usage

### CLI

```bash
# Run with default settings (5 iterations, gpt-4o-mini)
autoagentlab run qa

# Custom iterations and model
autoagentlab run qa --iterations 10 --model gpt-4o

# Custom initial prompt
autoagentlab run qa --prompt "You are a trivia expert. Be precise."

# List available benchmarks
autoagentlab list
```

### Python API

```python
from autoagentlab import Agent, ExperimentLoop

# Define your tasks
tasks = [
    ["What is 15 * 13?", "195"],
    ["What is the derivative of x^2?", "2x"],
    ["What is 2^10?", "1024"],
]

# Create an agent
agent = Agent("You are a math tutor. Answer concisely.")

# Run the experiment
loop = ExperimentLoop(agent, tasks, max_iterations=5)
history = loop.run()

# Check results
best = max(history, key=lambda r: r.accuracy)
print(f"Best: {best.accuracy * 100:.0f}% at iteration {best.iteration}")
```

---

## 🧩 Core Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `agent.py` | `Agent` | Wraps an LLM with a mutable system prompt |
| `evaluator.py` | `Evaluator` | Scores accuracy, records failure cases |
| `researcher.py` | `Researcher` | Analyzes failures, suggests improvements |
| `mutator.py` | `Mutator` | Rewrites the prompt based on suggestions |
| `loop.py` | `ExperimentLoop` | Orchestrates the full improvement cycle |
| `cli.py` | — | CLI interface (`autoagentlab run/list`) |

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Custom API endpoint (for compatible providers) |
| `AUTOAGENTLAB_MODEL` | `gpt-4o-mini` | Default LLM model |

> **Works with any OpenAI-compatible API** — use Ollama, Together AI, Groq, etc. by setting `OPENAI_BASE_URL`.

---

## 📝 Create Your Own Benchmark

1. Create a JSON file in `benchmarks/`:

```json
[
  ["Your question here?", "expected answer"],
  ["Another question?", "another answer"]
]
```

2. Run it:

```bash
autoagentlab run your_benchmark_name
```

---

## 🌟 Roadmap

- [ ] **Tool Discovery** — agents auto-discover and use tools (calculator, search, Python)
- [ ] **Workflow Mutation** — evolve multi-step reasoning chains
- [ ] **Agent Lineage** — track evolution history as a tree
- [ ] **Multi-objective** — optimize for accuracy + cost + latency
- [ ] **Population-based** — evolve a population of agents in parallel

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**AutoAgentLab** — *Let your agents do the research.* 🧬

</div>
