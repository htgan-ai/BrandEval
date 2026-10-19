"""Lightweight LLM interface and Gemini chat adapter

Provides a minimal `LLMInterface` used by pipeline nodes (Strategy/Summary)
and a Gemini HTTP adapter that wraps `brandeval.gemini_data.run`.

The adapter exposes a single `generate(messages, response_format, ...)` API
which accepts a list of role-content dicts and returns either text or JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import json

try:
    # Local lightweight Gemini HTTP wrapper
    from brandeval.gemini_data import run as gemini_run  # type: ignore
except Exception:
    gemini_run = None


class LLMInterface:
    """Minimal interface expected by pipeline nodes.

    Implementations must provide `generate(messages, response_format, ...)`.
    - messages: list of {"role": "system"|"user"|"assistant", "content": str}
    - response_format: "text" or "json"
    """

    def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: str = "text",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Union[str, Dict[str, Any]]:
        raise NotImplementedError


class GeminiChatLLM(LLMInterface):
    """Gemini chat adapter wrapping `gemini_data.run`.

    Since `gemini_data.run(user_input, history)` only accepts a single user
    input string and uses an internal (empty) system prompt, we collapse the
    provided messages into one concatenated string.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name

    def _collapse_messages(self, messages: List[Dict[str, str]]) -> str:
        system_parts: List[str] = []
        user_parts: List[str] = []
        other_parts: List[str] = []
        for m in messages or []:
            role = str(m.get("role", "user")).lower()
            content = str(m.get("content", ""))
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            else:
                other_parts.append(f"[{role}] {content}")
        # Place system first, then user, then any others
        combined = []
        if system_parts:
            combined.append("\n".join(system_parts))
        if user_parts:
            combined.append("\n".join(user_parts))
        if other_parts:
            combined.append("\n".join(other_parts))
        return "\n\n".join(combined).strip()

    def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: str = "text",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Union[str, Dict[str, Any]]:
        if gemini_run is None:
            # No backend available; return minimal fallbacks
            return {} if response_format == "json" else ""

        prompt_text = self._collapse_messages(messages)
        try:
            history, text = gemini_run(prompt_text, [], model_name=self.model_name)
        except Exception:
            # Backend call failed
            return {} if response_format == "json" else ""

        if response_format == "json":
            # Try direct parse; if fails, attempt to extract JSON object substring
            try:
                return json.loads(text)
            except Exception:
                # Heuristic extraction of the first JSON object
                try:
                    import re
                    match = re.search(r"\{[\s\S]*\}", text)
                    if match:
                        return json.loads(match.group(0))
                except Exception:
                    pass
                return {}
        # Default: return text
        return str(text or "")
