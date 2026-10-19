from __future__ import annotations

from typing import Optional, Any

from brandeval.pipeline.nodes.base_node import BaseNode, SandboxState


class SummaryNode(BaseNode):
    """t.4 Text summary generation for the step.

    Produces a short Markdown summary from metrics and simulation outputs.
    Uses LLM if available, otherwise a deterministic template.
    """

    name = "SummaryNode"

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def _llm_summary(self, state: SandboxState) -> Optional[str]:
        if not self.llm:
            return None
        try:
            content = {
                "t": state.step,
                "metrics": state.metrics,
                "strategy": state.final_strategy,
                "signals": state.simulation_outputs,
            }
            out = self.llm.generate(
                messages=[
                    {"role": "system", "content": "Summarize step results in concise Markdown."},
                    {"role": "user", "content": f"Summarize: {content}"},
                ],
                response_format="text",
            )
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception:
            pass
        return None

    def _fallback_summary(self, state: SandboxState) -> str:
        return (
            f"# Step {state.step} Summary\n\n"
            f"- Strategy: {state.final_strategy}\n"
            f"- Key metrics: {list(state.metrics.keys())}\n"
            f"- Signals: t={state.simulation_outputs.get('t')}\n"
        )

    def run(self, state: SandboxState, **kwargs) -> SandboxState:
        summary = self._llm_summary(state) or self._fallback_summary(state)
        state.summary = summary
        state.logs.append(f"[SummaryNode] t={state.step} produced summary length={len(summary)}")
        return state