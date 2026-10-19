from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from brandeval.pipeline.nodes.base_node import SandboxState
from brandeval.pipeline.nodes.strategy_node import StrategyNode
from brandeval.pipeline.nodes.dialogue_node import DialogueNode
from brandeval.pipeline.nodes.simulation_node import SimulationNode
from brandeval.pipeline.nodes.metrics_node import MetricsNode
from brandeval.pipeline.nodes.summary_node import SummaryNode


class Orchestrator:
    """Coordinates nodes to execute a complete time-step (t) sequence.

    Sequence per t:
      1) StrategyNode: build briefs, proposals, aggregate final strategy.
      2) SimulationNode: apply strategy, run one model step, capture signals.
      3) MetricsNode: compute metrics from the model.
      4) SummaryNode: produce a text summary (Markdown) for reporting.
    """

    def __init__(
        self,
        strategy: StrategyNode,
        simulation: SimulationNode,
        metrics: MetricsNode,
        summary: SummaryNode,
        output_dir: Optional[str] = None,
        dialogue: Optional[DialogueNode] = None,
    ):
        self.strategy = strategy
        self.simulation = simulation
        self.metrics = metrics
        self.summary = summary
        self.dialogue = dialogue
        self.output_dir = Path(output_dir or "./output/sandbox")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        return self.simulation.model

    def run_step(self, state: SandboxState) -> SandboxState:
        # t.0 Dialogue (optional)
        if self.dialogue is not None:
            state = self.dialogue.run(state)
        # t.1 Strategy
        state = self.strategy.run(state)
        # t.2 Simulation
        state = self.simulation.run(state)
        # t.3 Metrics
        state = self.metrics.run(state)
        # t.4 Summary
        state = self.summary.run(state)
        return state

    def persist(self, state: SandboxState, tag: Optional[str] = None) -> Path:
        tag = tag or f"t{state.step}"
        run_dir = self.output_dir / tag
        run_dir.mkdir(exist_ok=True)
        def _round4(obj):
            if isinstance(obj, float):
                return round(obj, 4)
            if isinstance(obj, dict):
                return {k: _round4(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_round4(x) for x in obj]
            return obj
        # Save JSON outputs
        (run_dir / "state.json").write_text(
            json.dumps(
                {
                    "step": state.step,
                    "background": state.background_info,
                    "dialogue": state.agent_communications,
                    "briefs": state.agent_briefs,
                    "proposals": state.agent_proposals,
                    "proposal_validations": state.proposal_validations,
                    "strategy": state.final_strategy,
                    "simulation": _round4(state.simulation_outputs),
                    "metrics": _round4(state.metrics),
                    "logs": state.logs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # Save Markdown summary
        (run_dir / "summary.md").write_text(state.summary, encoding="utf-8")
        # Save separate JSON files for process auditing (step three:全过程数据记录)
        try:
            (run_dir / "agent_communications.json").write_text(
                json.dumps(state.agent_communications, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            (run_dir / "proposal_validations.json").write_text(
                json.dumps(state.proposal_validations, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            model = getattr(self, "model", None)
            if model is not None:
                hist = getattr(model, "history", [])
                if hist:
                    (run_dir / "history.json").write_text(
                        json.dumps(_round4(hist[-1]), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                records = list(getattr(model, "jsonl_records", []) or [])
                if records:
                    (run_dir / "metrics.json").write_text(
                        json.dumps(_round4(records[-1].get("market_metrics", {})), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    try:
                        posts = list(records[-1].get("consumer_generated_content", []) or [])
                        norm = []
                        import re as _re
                        import ast as _ast
                        for p in posts:
                            q = dict(p)
                            c = q.get("content")
                            if isinstance(c, str) and "'content'" in c:
                                try:
                                    parsed = None
                                    if c.strip().startswith("{"):
                                        try:
                                            import json as _json
                                            parsed = _json.loads(c)
                                        except Exception:
                                            parsed = _ast.literal_eval(c)
                                    else:
                                        parsed = _ast.literal_eval(c)
                                    if isinstance(parsed, dict):
                                        q["content"] = str(parsed.get("content") or parsed.get("text") or q.get("content"))
                                        if parsed.get("brand_id") is not None:
                                            q["brand_id"] = parsed.get("brand_id")
                                except Exception:
                                    m = _re.search(r"'content'\s*:\s*'([^']*)'", c)
                                    if m:
                                        q["content"] = m.group(1)
                                    m2 = _re.search(r"'brand_id'\s*:\s*'([^']*)'", c)
                                    if m2:
                                        q["brand_id"] = m2.group(1)
                            norm.append(q)
                        (run_dir / "posts.json").write_text(
                            json.dumps(norm, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        (run_dir / "posts.json").write_text(
                            json.dumps(records[-1].get("consumer_generated_content", []), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    try:
                        from pathlib import Path as _P
                        jl = "\n".join([json.dumps(r, ensure_ascii=False) for r in records])
                        (_P(run_dir) / "jsonl_records.jsonl").write_text(jl, encoding="utf-8")
                    except Exception:
                        pass
                prev_msgs = list(getattr(model, "_message_buffer_prev", []) or [])
                curr_msgs = list(getattr(model, "_message_buffer_current", []) or [])
                try:
                    if (not curr_msgs) and records:
                        posts = list(records[-1].get("consumer_generated_content", []) or [])
                        synthesized = []
                        for p in posts:
                            synthesized.append({
                                "sender_id": p.get("authorId"),
                                "sender_role": "consumer",
                                "message_type": "ugc_post",
                                "content": p.get("content"),
                            })
                        curr_msgs = synthesized
                except Exception:
                    pass
                (run_dir / "messages_prev.json").write_text(
                    json.dumps(_round4(prev_msgs), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "messages_current.json").write_text(
                    json.dumps(_round4(curr_msgs), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass
        return run_dir