from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


@dataclass
class SandboxState:
    """Shared state passed between nodes in the pipeline.

    - Holds background info, strategies, simulation results, metrics, and summary.
    - Designed for reproducible, script-only runs.
    """

    step: int = 0
    background_info: Dict[str, Any] = field(default_factory=dict)

    # coordination & strategy
    agent_briefs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    agent_proposals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_strategy: Dict[str, Any] = field(default_factory=dict)

    # simulation & metrics
    simulation_outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # text summary
    summary: str = ""
    logs: List[str] = field(default_factory=list)

    # agent dialogue & validations (per timestep)
    agent_communications: List[Dict[str, Any]] = field(default_factory=list)
    proposal_validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BaseNode:
    """Abstract base node.

    Each node implements `run(state)` returning the updated `SandboxState`.
    """

    name: str = "BaseNode"

    def run(self, state: SandboxState, **kwargs) -> SandboxState:
        raise NotImplementedError