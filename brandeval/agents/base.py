"""Minimal BrandEvalAgent base class

Provides a lightweight base with messaging, neighbor retrieval, and memory
utilities used by Brand, Consumer, and strategic agents.

This avoids heavy dependencies and lets the sandbox run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import hashlib


class BrandEvalAgent:
    """Lightweight base for agents in the BrandEval sandbox.

    - Holds references to the model and role.
    - Provides `send_message`, `receive_message`, and `get_neighbors` helpers.
    - Stores simple in-agent memory logs via `_store_in_memory`.
    """

    def __init__(self, model: Any, role: str = "agent") -> None:
        self.model = model
        self.role = str(role or "agent")

        # Per-role unique id assignment
        counters = getattr(model, "_agent_role_counters", None)
        if counters is None:
            counters = {}
            try:
                setattr(model, "_agent_role_counters", counters)
            except Exception:
                pass
        nxt = counters.get(self.role, 1)
        self.unique_id = f"{self.role}_{nxt}"
        counters[self.role] = nxt + 1

        # Optional fields used by messaging/network spaces.
        self.pos: Optional[int] = None
        self.message_queue: List[Dict[str, Any]] = []
        self.memory: List[Dict[str, Any]] = []

        # Reuse model RNG if available for deterministic runs.
        self.random = getattr(model, "random", None)

        # Influence is used in probabilistic push; subclasses can override.
        self.influence: float = 0.0

    # ===== Messaging helpers =====
    def send_message(self, recipients: List[Any], message_type: str, content: Any) -> None:
        """Send a message into the platform bus.

        - If `recipients` is empty, the platform records it and handles distribution
          at t+1 (official/ugc posts).
        - If recipients are provided, each gets an immediate `receive_message` call.
        """
        msg = {
            "sender_id": getattr(self, "unique_id", None),
            "sender_role": getattr(self, "role", "agent"),
            "message_type": str(message_type or "message"),
            "content": content,
        }

        # Record to the platform bus (used for posts/JSONL/export/distribution)
        try:
            if hasattr(self.model, "record_message"):
                self.model.record_message(msg)
        except Exception:
            pass

        # Immediate direct delivery to provided recipients
        for r in recipients or []:
            try:
                r.receive_message(msg)
            except Exception:
                # Best-effort delivery; platform bus still captured
                pass

    def receive_message(self, message: Dict[str, Any]) -> None:
        """Default message intake: append to queue for later processing."""
        try:
            self.message_queue.append(message)
        except Exception:
            # Ensure no hard failure blocks simulation
            pass

    # ===== Network helpers =====
    def get_neighbors(self, radius: int = 1, include_self: bool = False) -> List[Any]:
        """Return neighbor agents from the model's Network space if present."""
        space = getattr(self.model, "space", None)
        if space is None:
            return []
        try:
            node_id = getattr(self, "pos", None)
            # If agent hasn't been placed, attempt a simple placement using unique index.
            if node_id is None:
                # Place in a deterministic node based on id hash, if supported.
                n_nodes = max(1, len(getattr(self.model.network, "nodes", [])))
                # Python's built-in hash is process-randomized; use a stable
                # digest so seeded runs place agents identically across runs.
                digest = hashlib.sha256(str(self.unique_id).encode("utf-8")).digest()
                node_id = int.from_bytes(digest[:8], "big") % n_nodes
                try:
                    space.place_agent(self, node_id)
                    self.pos = node_id
                except Exception:
                    # If placement fails, skip neighbor lookup.
                    return []
            return list(space.get_neighbors(node_id, include_center=include_self, radius=radius))
        except Exception:
            return []

    # ===== Memory helpers =====
    def _store_in_memory(
        self,
        item: Dict[str, Any],
        importance: float = 0.5,
        memory_type: str = "note",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a structured memory item for later inspection.

        The sandbox does not implement retrieval; nodes or agents can read
        the list directly when needed.
        """
        rec = {
            "importance": float(importance),
            "type": str(memory_type or "note"),
            "tags": list(tags or []),
            "payload": item,
        }
        try:
            self.memory.append(rec)
        except Exception:
            pass
