from __future__ import annotations

from typing import Any, Dict, List, Optional

from brandeval.pipeline.nodes.base_node import BaseNode, SandboxState


class DialogueNode(BaseNode):
    """t.0 Agent dialogue & coordination stage.

    - Simulates multi-round exchange between Brand and Competitor (or counterfactual actor).
    - Uses provided LLM to generate structured dialogue when available.
    - Falls back to heuristic messages to ensure run stability.
    - Persists full message list into `state.agent_communications` for this timestep.
    """

    name = "DialogueNode"

    def __init__(self, llm: Optional[Any] = None, rounds: int = 3):
        self.llm = llm
        self.rounds = max(1, int(rounds))

    def _llm_dialogue(self, bg: Dict[str, Any], rounds: int) -> List[Dict[str, Any]]:
        if not self.llm:
            return []
        try:
            brand_name = str(bg.get("brand_name") or (bg.get("brand_keywords") or [None])[0] or bg.get("brand_id") or "品牌")
            comp_kw = (bg.get("competitor_keywords") or [])
            competitor_name = str(comp_kw[0] if comp_kw else "竞品")
            terms = []
            for k in (bg.get("brand_keywords") or []) + (bg.get("competitor_keywords") or []):
                if k and k not in terms:
                    terms.append(k)
            topic = str(bg.get("dominant_topic") or "")
            scene = str(bg.get("exam_text") or "")
            prompt = (
                "你是数字沙盘的对话协调者，请用中文生成品牌与竞品的正式公告/宣传式对话。"
                "严格返回一个 JSON 对象，包含键 'messages'；其值是对象列表，每项包含："
                "'t'(从1开始的轮次序号，整数)、'speaker'(brand 或 competitor)、'content'(中文)。"
                "风格要求：\n"
                "- 品牌/竞品：正式公告或宣传口吻，克制、专业、信息清晰；不得使用占位符、不得口头化语气；避免感叹与俚语；每条不超过两句。\n"
                "- 消费者内容不在本对话中生成。\n"
                "语境：主题='" + topic + "'；场景='" + scene + "'；品牌=" + brand_name + "；竞品=" + competitor_name + "；术语=" + ",".join(terms) + "。\n"
                "定位：危机沟通与技术迭代；品牌强调透明修复与进度；竞品强调主动召回与可靠性。\n"
                "总轮次=" + str(rounds) + "。"
            )
            out = self.llm.generate(
                messages=[
                    {"role": "system", "content": "只输出严格 JSON；不得出现占位符或英文字母代词；所有句子为中文、口语化。"},
                    {"role": "user", "content": prompt},
                ],
                response_format="json",
            )
            if isinstance(out, dict) and isinstance(out.get("messages"), list):
                msgs = []
                for m in out["messages"]:
                    try:
                        t_i = int(m.get("t", 1))
                        spk = str(m.get("speaker", "brand"))
                        txt = str(m.get("content", ""))
                        def _sanitize(s: str) -> str:
                            repl = ["哎呀", "哼", "哦", "啊", "吧", "呢", "嘛", "哈", "呃", "唉"]
                            for r in repl:
                                s = s.replace(r, "")
                            s = s.replace("！", "。").replace("!", "。")
                            s = s.replace("??", "。").replace("?", "。")
                            s = s.replace("XX", "")
                            s = s.replace("[", "").replace("]", "")
                            s = s.strip()
                            if s and s[-1] not in "。；：":
                                s = s + "。"
                            return s
                        msgs.append({
                            "t": t_i,
                            "speaker": spk,
                            "content": _sanitize(txt),
                        })
                    except Exception:
                        continue
                return msgs
        except Exception:
            pass
        return []

    def _heuristic_dialogue(self, bg: Dict[str, Any], rounds: int) -> List[Dict[str, Any]]:
        return []

    def run(self, state: SandboxState, **kwargs) -> SandboxState:
        bg = state.background_info or {}
        messages = self._llm_dialogue(bg, self.rounds) or self._heuristic_dialogue(bg, self.rounds)
        # Attach timestep metadata and persist into state
        enriched = [{"timestep": state.step, **m} for m in messages]
        state.agent_communications = enriched
        state.logs.append(f"[DialogueNode] t={state.step} messages={len(enriched)}")
        return state