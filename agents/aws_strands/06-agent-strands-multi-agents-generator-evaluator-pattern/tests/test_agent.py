import pytest
from pathlib import Path


# ── mem helpers ────────────────────────────────────────────────────
def test_mem_init_creates_all_sections(tmp_path, monkeypatch):
    import agent
    monkeypatch.setattr(agent, "MEM", tmp_path / "memory.md")
    agent.mem_init("AI Agents on AWS")
    text = (tmp_path / "memory.md").read_text()
    for section in ["## Plan", "## Research", "## Sources",
                    "## Critiques", "## Route Log", "## Log"]:
        assert section in text, f"Missing section: {section}"
    assert "AI Agents on AWS" in text


def test_mem_append_and_read(tmp_path, monkeypatch):
    import agent
    monkeypatch.setattr(agent, "MEM", tmp_path / "memory.md")
    agent.mem_init("Test Topic")
    agent.mem_append("Plan", "Research angle: focus on 2024 updates")
    text = agent.mem_read()
    assert "Research angle: focus on 2024 updates" in text


def test_mem_count_generator_runs_zero(tmp_path, monkeypatch):
    import agent
    monkeypatch.setattr(agent, "MEM", tmp_path / "memory.md")
    agent.mem_init("Test Topic")
    assert agent.mem_count_generator_runs() == 0


def test_mem_count_generator_runs_increments(tmp_path, monkeypatch):
    import agent
    monkeypatch.setattr(agent, "MEM", tmp_path / "memory.md")
    agent.mem_init("Test Topic")
    agent.mem_append("Log", "- `10:00:00` generator_run\n")
    agent.mem_append("Log", "- `10:05:00` generator_run\n")
    assert agent.mem_count_generator_runs() == 2


# ── _parse_routing ─────────────────────────────────────────────────
class MockResult:
    def __init__(self, text): self.result = text


class MockState:
    def __init__(self, router_text=None):
        self.results = {}
        if router_text is not None:
            self.results["router"] = MockResult(router_text)


def test_parse_routing_generator():
    import agent
    state = MockState('Some text\nROUTING_DECISION: {"next": "generator", "reason": "no blog yet"}')
    assert agent._parse_routing(state) == "generator"


def test_parse_routing_evaluator():
    import agent
    state = MockState('ROUTING_DECISION: {"next": "evaluator", "reason": "blog ready"}')
    assert agent._parse_routing(state) == "evaluator"


def test_parse_routing_done():
    import agent
    state = MockState('ROUTING_DECISION: {"next": "done", "reason": "accepted"}')
    assert agent._parse_routing(state) == "done"


def test_parse_routing_no_router_defaults_generator():
    import agent
    assert agent._parse_routing(MockState()) == "generator"


def test_parse_routing_malformed_falls_back():
    import agent
    state = MockState("I think we should generate next.")
    assert agent._parse_routing(state) == "done"


def test_route_conditions():
    import agent
    gen_state  = MockState('ROUTING_DECISION: {"next": "generator", "reason": "x"}')
    eval_state = MockState('ROUTING_DECISION: {"next": "evaluator", "reason": "x"}')
    done_state = MockState('ROUTING_DECISION: {"next": "done", "reason": "x"}')
    assert agent._route_to_generator(gen_state) is True
    assert agent._route_to_generator(eval_state) is False
    assert agent._route_to_evaluator(eval_state) is True
    assert agent._route_to_evaluator(done_state) is False
