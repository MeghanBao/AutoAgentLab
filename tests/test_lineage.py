"""Tests for agent lineage tracking — no LLM calls required."""

from autoagentlab.agent import Agent


def test_agent_has_id():
    a = Agent("test prompt")
    assert isinstance(a.agent_id, str)
    assert len(a.agent_id) == 8


def test_agent_default_no_parent():
    a = Agent("test prompt")
    assert a.parent_id is None


def test_agent_default_generation_zero():
    a = Agent("test prompt")
    assert a.generation == 0


def test_clone_parent_id_matches():
    parent = Agent("original prompt")
    child = parent.clone("mutated prompt")
    assert child.parent_id == parent.agent_id


def test_clone_has_new_id():
    parent = Agent("original prompt")
    child = parent.clone("mutated prompt")
    assert child.agent_id != parent.agent_id


def test_clone_generation_increments():
    a = Agent("gen 0")
    b = a.clone("gen 1")
    c = b.clone("gen 2")
    assert a.generation == 0
    assert b.generation == 1
    assert c.generation == 2


def test_clone_inherits_model():
    a = Agent("prompt", model="gpt-4o")
    child = a.clone("new prompt")
    assert child.model == "gpt-4o"


def test_clone_inherits_tools_enabled():
    a = Agent("prompt", tools_enabled=True)
    child = a.clone("new prompt")
    assert child.tools_enabled is True


def test_unique_ids_across_agents():
    agents = [Agent(f"prompt {i}") for i in range(50)]
    ids = [a.agent_id for a in agents]
    assert len(set(ids)) == 50, "All agent IDs should be unique"
