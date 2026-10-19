from __future__ import annotations

from typing import Any, Dict, Optional, List
import math

from brandeval.pipeline.nodes.base_node import BaseNode, SandboxState


class StrategyNode(BaseNode):
    """t.1 Strategy generation and agent coordination proposals.

    - Accepts background info and builds briefs for each agent.
    - Calls LLM (if provided) to produce JSON strategies; otherwise uses a simple heuristic.
    - Aggregates proposals into `final_strategy` for the simulation.
    """

    name = "StrategyNode"

    def __init__(self, llm: Optional[Any] = None, agent_ids: Optional[list[str]] = None):
        self.llm = llm
        self.agent_ids = agent_ids or ["brand"]

    def _build_brief(self, state: SandboxState, agent_id: str) -> Dict[str, Any]:
        bg = state.background_info or {}
        brief = {
            "current_timestep": state.step,
            "dominant_topic": bg.get("dominant_topic", ""),
            "key_negative_posts": bg.get("key_negative_posts", []),
            "market_signals": bg.get("market_signals", {}),
            "constraints": bg.get("constraints", {}),
        }
        return brief

    def _llm_strategy(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return {}
        try:
            # Expecting the LLMInterface to return a structured JSON-like dict.
            # Minimal prompt contract: model-side handles system/user assembly.
            output = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a marketing strategy planner returning JSON."},
                    {"role": "user", "content": f"Brief: {brief}. Return a JSON with keys: seed_selection, budget, channels, messaging."},
                ],
                response_format="json",
            )
            if isinstance(output, dict):
                return output
        except Exception:
            pass
        return {}

    def _heuristic_strategy(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        # Simple defaults to keep the pipeline runnable
        topic = brief.get("dominant_topic") or "brand_awareness"
        return {
            "seed_selection": {"method": "high_degree", "count": 20},
            "budget": {"total": 10000, "allocation": {"social": 0.5, "community": 0.3, "ads": 0.2}},
            "channels": ["social", "community"],
            "messaging": {"theme": topic, "tone": "transparent"},
        }

    def _aggregate(self, proposals: Dict[str, Dict[str, Any]], validations: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Prefer first VALID proposal; otherwise choose first non-empty; else fallback default
        if isinstance(validations, dict) and validations:
            # Return first valid; if none valid, fall back to default schema
            for aid, p in proposals.items():
                v = validations.get(aid) or {}
                if p and bool(v.get("valid")):
                    return p
            return {
                "seed_selection": {"method": "random", "count": 10},
                "budget": {"total": 5000, "allocation": {"social": 0.5, "community": 0.5}},
                "channels": ["social"],
                "messaging": {"theme": "default", "tone": "neutral"},
            }
        # No validations provided: choose first non-empty
        for _, p in proposals.items():
            if p:
                return p
        return {
            "seed_selection": {"method": "random", "count": 10},
            "budget": {"total": 5000, "allocation": {"social": 0.5, "community": 0.5}},
            "channels": ["social"],
            "messaging": {"theme": "default", "tone": "neutral"},
        }

    def _validate_strategy(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Strict schema validation with type checks and error reporting.

        Required keys:
          - seed_selection: {method: str, count: int}
          - budget: {total: number, allocation: dict str->number}
          - channels: list[str]
          - messaging: {theme: str, tone: str}
        """
        errors: List[str] = []

        def finite_number(value: Any) -> bool:
            # bool is an int subclass but is not a valid budget/count value.
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

        try:
            if not isinstance(proposal, dict):
                errors.append("proposal_not_dict")
                return {"valid": False, "errors": errors}
            # seed_selection
            ss = proposal.get("seed_selection")
            if not isinstance(ss, dict):
                errors.append("seed_selection_missing_or_not_dict")
            else:
                if not isinstance(ss.get("method"), str):
                    errors.append("seed_selection.method_not_str")
                if not finite_number(ss.get("count")):
                    errors.append("seed_selection.count_not_number")
                elif float(ss.get("count")) < 0 or float(ss.get("count")) % 1:
                    errors.append("seed_selection.count_not_nonnegative_integer")
            # budget
            bd = proposal.get("budget")
            if not isinstance(bd, dict):
                errors.append("budget_missing_or_not_dict")
            else:
                if not finite_number(bd.get("total")):
                    errors.append("budget.total_not_number")
                elif float(bd.get("total")) < 0:
                    errors.append("budget.total_negative")
                alloc = bd.get("allocation")
                if not isinstance(alloc, dict) or not all(isinstance(k, str) for k in alloc.keys()):
                    errors.append("budget.allocation_not_dict_str_keys")
                else:
                    if not all(finite_number(v) for v in alloc.values()):
                        errors.append("budget.allocation_values_not_number")
                    elif any(float(v) < 0 for v in alloc.values()):
                        errors.append("budget.allocation_negative")
                    elif not alloc or not math.isclose(sum(float(v) for v in alloc.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
                        errors.append("budget.allocation_not_normalized")
            # channels
            ch = proposal.get("channels")
            if not isinstance(ch, list) or not all(isinstance(x, str) for x in ch):
                errors.append("channels_not_list_str")
            # messaging
            msg = proposal.get("messaging")
            if not isinstance(msg, dict):
                errors.append("messaging_missing_or_not_dict")
            else:
                if not isinstance(msg.get("theme"), str):
                    errors.append("messaging.theme_not_str")
                if not isinstance(msg.get("tone"), str):
                    errors.append("messaging.tone_not_str")
        except Exception:
            errors.append("validation_exception")
        return {"valid": len(errors) == 0, "errors": errors}

    def run(self, state: SandboxState, **kwargs) -> SandboxState:
        # Build briefs
        briefs: Dict[str, Dict[str, Any]] = {}
        proposals: Dict[str, Dict[str, Any]] = {}
        validations: Dict[str, Dict[str, Any]] = {}
        for aid in self.agent_ids:
            brief = self._build_brief(state, aid)
            briefs[aid] = brief
            proposal = self._llm_strategy(brief) or self._heuristic_strategy(brief)
            proposals[aid] = proposal
            validations[aid] = self._validate_strategy(proposal)

        state.agent_briefs = briefs
        state.agent_proposals = proposals
        state.proposal_validations = validations
        state.final_strategy = self._aggregate(proposals, validations)
        state.logs.append(
            f"[StrategyNode] t={state.step} aggregated strategy (valid-first, fallback-default if invalid): {state.final_strategy}"
        )
        return state
