"""Strategic Agent (w/o SR): single-action ablation of the brand policy.

Table 6 of the paper compares the SR-guided Strategic Agent with a variant
that skips the Strategic Rationale workflow (no verifiable goal, no
stakeholder-state summary, no candidate planning).  This class keeps the
same observation, evidence gate, and publication path as ``BrandAgent`` and
only changes the prompt and the output schema.  Select it with
``--brand-policy no_sr`` in the sandbox CLI.
"""

from typing import Any, Dict, List

import json

from brandeval.agents.brand_agent import BrandAgent


BRAND_AGENT_NO_SR_PROMPT_TEMPLATE = (
    """
一、品牌Agent Prompt
# 角色：你是一个世界顶级的品牌战略官 (BrandAgent)。
# 核心规则：你的所有决策都必须服务于你的【品牌身份】，并旨在【优化沙盘指标】，同时要考虑行动的【必要性】和【成本效益】。

# 【1. 品牌身份注入】
# (此部分由系统在初始化时固定)
{
  "brand_name": "{brand_name}",
  "product_name": "{product_name}",
  "intended_positioning": {intended_positioning_list},
  "intended_personality": {intended_personality_list},
  "intended_relationship": {intended_relationship_list}
}

# 【2. 市场情报简报】
# (此部分由“平台环境”在 t.1 阶段动态填充)
{
  "current_timestep": {t},
  "dominant_topic": "{dominant_topic_string}",
  "key_negative_posts": {key_posts_list_of_strings},
  "metrics": {metrics_json_object},
  "competitor_last_move": "{competitor_move_string}"
}

# 【3. 核心决策任务：思考链】
# (此部分是LLM被要求的思考过程)
# 请你基于【1. 品牌身份】和【2. 市场情报】，严格遵循以下思考链：

## 步骤 3.1: [情境分析]
* **首先，评估【市场情报简报】。** 当前最紧迫的威胁/机遇是什么？它直接冲击了我的哪个CBBE维度？
* **【关键判断】** 这个问题是**严重**且**紧急**的吗？还是仅仅是微小的波动或噪音？ (例如：BRI只是略低于目标0.6，还是暴跌？NPS是否大幅下滑？新话题是致命危机，还是小范围抱怨？)
* **【特别注意】** 
    1. 请敏锐识别任何可能影响**购买决策**的关键负面信号（如价格过高、质量缺陷、虚假宣传等）。
    2. **检查对话是否陷入僵局**：如果消费者一直在重复质疑同一个点（如“空话”、“没数据”），你必须改变策略，提供**新**的信息或证据。

## 步骤 3.2: [身份对齐]
* 根据我的【品牌身份】，我对当前情境（无论严重与否）的**第一反应原则**应该是什么？ (例如：“坦诚沟通”、“快速响应”或“保持稳健”)
* **【主动性要求】** 作为顶级品牌，必须保持持续的市场声量。即使没有危机，也应主动发布品牌宣传、产品亮点或价值观故事，以维持高SoV和BRI。

## 步骤 3.3: [直接行动生成]
* 基于 3.1 的严重性判断和 3.2 的原则，**直接生成 1 个**你认为最优的行动方案。
* **【重要 - 拒绝空话】** 
    1. **严禁**使用通用的公关辞令（如“我们重视您的反馈”）。
    2. 必须提供**实质性内容**：引用数据、案例或直接回应争议。
    3. 你的目标是**一步到位**，生成最能优化 BRI/NPS 的内容。

## 步骤 3.4: [最终确认]
* 确认该行动方案符合品牌身份，准备输出。

# 【4. 最终行动与白盒原理】
# (此部分是LLM必须严格遵循的最终输出格式)
# 请将你 3.1 至 3.4 的思考链，总结为以下JSON格式。

{
  "rationale_summary": "...",
  # (LLM 填入对 3.1 (严重性判断) 和 3.2 的总结。) 
  "expected_outcome": "...",
  # (LLM 填入对该单一行动方案的预期效果预测。) 
  "risks_considered": "...",
  # (LLM 填入执行这唯一方案可能带来的风险，) 
  "final_action_content": "...",
  # (LLM 填入 3.3 中最优选项的具体文案。必须是**模拟的社交媒体帖子内容**，用于发布给消费者看。
  #  **关键要求**：
  #  1. 不要包含策略描述或选项编号。
  #  2. **严禁**填入 "No significant action taken."。
  #  3. **拒绝复读**：不要重复之前的口号。
  #  4. **具体化**：针对当前的负面帖子（key_negative_posts），点对点地进行有力回击或解释，言之有物。)
  "claimed_evidence": ["..."],
  # (List of Evidence IDs referenced in the content, e.g. ["EDR-Log-123", "Recall-Batch-992"]. 
  #  Leave empty [] if no specific evidence is cited.)
  "predicted_impact": {
      "capability": 0.1,
      "value": 0.1,
      "emotion": 0.1
  }
  # (LLM 预估该行动对消费者感知的量化影响。范围 [-1.0, 1.0]。)
}
"""
).strip()


class BrandAgentNoSR(BrandAgent):
    """Brand policy without the Strategic Rationale workflow."""

    def render_brand_agent_prompt(
        self,
        t: int,
        dominant_topic: str,
        key_negative_posts: List[str],
        metrics: Dict[str, Any],
        competitor_move_string: str,
    ) -> str:
        tpl = BRAND_AGENT_NO_SR_PROMPT_TEMPLATE
        tpl = tpl.replace("{intended_positioning_list}", json.dumps(self.identity.get("intended_positioning", []), ensure_ascii=False))
        tpl = tpl.replace("{intended_personality_list}", json.dumps(self.identity.get("intended_personality", []), ensure_ascii=False))
        tpl = tpl.replace("{intended_relationship_list}", json.dumps(self.identity.get("intended_relationship", []), ensure_ascii=False))
        tpl = tpl.replace("{brand_name}", self.brand_name)
        tpl = tpl.replace("{product_name}", self.product_name)
        tpl = tpl.replace("{t}", str(t))
        tpl = tpl.replace("{dominant_topic_string}", str(dominant_topic or ""))
        tpl = tpl.replace("{key_posts_list_of_strings}", json.dumps(key_negative_posts or [], ensure_ascii=False))
        tpl = tpl.replace("{metrics_json_object}", json.dumps(metrics or {}, ensure_ascii=False))
        tpl = tpl.replace("{competitor_move_string}", str(competitor_move_string or ""))
        return tpl

    def _decision_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rationale_summary": {"type": "string"},
                "expected_outcome": {"type": "string"},
                "risks_considered": {"type": "string"},
                "final_action_content": {"type": "string"},
                "claimed_evidence": {"type": "array", "items": {"type": "string"}},
                "predicted_impact": {
                    "type": "object",
                    "properties": {
                        "capability": {"type": "number"},
                        "value": {"type": "number"},
                        "emotion": {"type": "number"},
                    },
                    "required": ["capability", "value", "emotion"],
                },
            },
            "required": ["rationale_summary", "expected_outcome", "risks_considered", "final_action_content", "predicted_impact"],
            "additionalProperties": False,
        }

    def _normalize_llm_json(self, result: Any) -> Dict[str, Any]:
        required = ["rationale_summary", "expected_outcome", "risks_considered", "final_action_content"]
        if isinstance(result, dict) and all(k in result for k in required):
            result.setdefault("predicted_impact", {"capability": 0.0, "value": 0.0, "emotion": 0.0})
            evidence = result.get("claimed_evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence] if evidence.strip() else []
            result["claimed_evidence"] = evidence if isinstance(evidence, list) else []
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return self._normalize_llm_json(parsed)
            except Exception:
                pass
        return {
            "rationale_summary": "API call failed, please check API",
            "expected_outcome": "",
            "risks_considered": "",
            "final_action_content": "",
            "predicted_impact": {"capability": 0.0, "value": 0.0, "emotion": 0.0},
            "claimed_evidence": [],
        }

    def _heuristic_action(self, dominant_topic: str, key_negative_posts: List[str]) -> Dict[str, Any]:
        action = super()._heuristic_action(dominant_topic, key_negative_posts)
        for key in ("goal", "stakeholder_summary", "candidate_plans", "selected_plan"):
            action.pop(key, None)
        return action
