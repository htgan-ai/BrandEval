"""BrandAgent: KPI 触发 + CoT 决策架构

实现实验组品牌Agent，支持：
- 静态身份配置（定位、人格、关系）
- 动态KPI触发器（BRI/RNSR/新负面话题）
- CoT Prompt 构建与 LLM JSON 输出
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import random
import time
import os
import math

from brandeval.agents.base import BrandEvalAgent
from brandeval.llm.interface import LLMInterface
try:
    from brandeval.gemini_data import run as gemini_run  # type: ignore
except Exception:
    gemini_run = None


# 品牌Agent Prompt 模板（显式呈现，可用于LLM调用时渲染占位符）
BRAND_AGENT_PROMPT_TEMPLATE = (
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
* 只在当前公开证据支持的情况下提出回应；不得为了提高声量、SoV 或内部指标而制造信息或无必要地发帖。

## 步骤 3.3: [行动选项生成]
* 基于 3.1 的严重性判断和 3.2 的原则，生成 **2-3 个**合理的行动选项。
* **【重要 - 拒绝空话】** 
    1. **严禁**使用通用的公关辞令（如“我们重视您的反馈”、“我们将继续努力”）。
    2. 必须提供**实质性内容**：引用具体性能数据、用户真实案例、对比评测结果，或直接回应核心争议（如CEO事件）。
    3. 如果之前的策略是“情感感召”无效，尝试“理性数据”或“反讽幽默”等新角度。
* **【默认倾向】** 只有在公开证据和利益相关者需求支持时才发声；不得为了提高声量而发帖。

## 步骤 3.4: [多维影响评估]
* 逐一推演每个选项对沙盘指标（BRI/CBBE, NPS, Process Health, Competitor Impact）的长期影响。
* **【关键考量】** 评估每个行动的**潜在收益** vs. **潜在风险/成本**。 (例如：“发布声明”能否真的提升BRI/NPS？“不行动”会导致SoV下降和被竞品超越。)

## 步骤 3.5: [最终决策]
* 基于 3.4 的多维评估和成本效益分析，选择一个最优选项。
* 只有在极少数情况下（如必须静默期），才选择“不行动”。

# 【4. 最终行动与白盒原理】
# (此部分是LLM必须严格遵循的最终输出格式)
# 请将你 3.1 至 3.5 的思考链，总结为以下JSON格式。

{
  "goal": "...",
  "stakeholder_summary": "...",
  "candidate_plans": ["...", "...", "..."],
  "selected_plan": "...",
  "rationale_summary": "...",
  # (LLM 填入对 3.1 (严重性判断) 和 3.2 的总结。) 
  "expected_outcome": "...",
  # (LLM 填入对 3.4 中最优选项的推演结果总结。) 
  "risks_considered": "...",
  # (LLM 填入选择该选项的主要风险。) 
  "final_action_content": "...",
  # (LLM 填入 3.3 中最优选项的具体文案。必须是**模拟的社交媒体帖子内容**，用于发布给消费者看。
  #  **关键要求**：
  #  1. 不要包含策略描述或选项编号。
  #  2. **严禁**填入 "No significant action taken."。
  #  3. **拒绝复读**：不要重复之前的口号。
  #  4. **具体化**：针对当前公开信息，给出事实、时间线、责任边界或可核验的后续行动。)
  "claimed_evidence": "...",
  # (Evidence identifier registered in the current evidence store, or an empty string.)
  "predicted_impact": {
      "capability": 0.1,
      "value": 0.1,
      "emotion": 0.1
  }
  # (LLM 预估该行动对消费者感知的量化影响。范围 [-1.0, 1.0]。)
}
"""
).strip()


class BrandAgent(BrandEvalAgent):
    """品牌Agent（实验组）：基于BrandSRD范式的动态决策架构。

    - 通过 KPI 警报触发完整的 LLM 决策流程（Prompt C）。
    - 否则采取默认行动（继续观察）。
    """

    def __init__(
        self,
        model,
        llm: Optional[LLMInterface] = None,
        identity: Optional[Dict[str, List[str]]] = None,
        name: str = "BrandAgent",
        brand_name: str = "My Brand",
        product_name: str = "My Product",
        goal_bri_floor: float = 60.0,
        goal_nps_floor: float = 0.2,
    ):
        super().__init__(model, role="brand")
        self.name = name
        self.brand_name = brand_name
        self.product_name = product_name
        self.llm = llm
        # 静态身份配置
        self.identity = identity or {
            "intended_positioning": ["quality", "innovation"],
            "intended_personality": ["transparent", "reliable"],
            "intended_relationship": ["customer-first", "long-term"]
        }
        # 目标KPI
        self.goal_bri_floor = self._normalize_bri_threshold(goal_bri_floor)
        self.goal_nps_floor = self._normalize_nps_threshold(goal_nps_floor)

        # 主题追踪
        self._seen_topics = set()
        self._last_dominant_topic: Optional[str] = None
        # 占位销量字段，避免数据收集阶段访问缺失属性
        self.sales: float = 0.0
        self.reputation: float = 0.0

    @staticmethod
    def _normalize_bri_threshold(value: float) -> float:
        """Accept BRI floors on [0, 1] and convert them to the point scale."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("goal_bri_floor must be finite")
        return value * 100.0 if abs(value) <= 1.0 else value

    @staticmethod
    def _normalize_nps_threshold(value: float) -> float:
        """Accept NPS floors in percentage points and convert them to [-1, 1]."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("goal_nps_floor must be finite")
        return value / 100.0 if abs(value) > 1.0 else value

    # ========== Gemini 占位适配器 ==========
    class GeminiLLM:
        """轻量Gemini占位实现：提供 generate_with_json / generate_text。"""

        def __init__(self, model_name: str = "gemini-1.5-pro-latest"):
            self.model_name = model_name

        def generate_with_json(self, prompt: str, json_schema: Dict[str, Any], temperature: float = 0.2, max_tokens: int = 1200) -> Dict[str, Any]:
            if (
                gemini_run is None
                or os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", "1") != "0"
                or str(self.model_name).strip().lower() in {"offline", "heuristic", "none"}
            ):
                raise RuntimeError("Gemini API not initialized")
            user_input = (
                f"{prompt}\n\n"
                "请严格只输出一个JSON对象，并满足如下模式要求（不要输出其他文字）：\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
            try:
                history, text = gemini_run(user_input, [], model_name=self.model_name)
                # Clean markdown code blocks
                text = str(text).strip()
                if text.startswith("```"):
                    import re
                    text = re.sub(r"^```\w*\s*|```\s*$", "", text, flags=re.MULTILINE).strip()
                return json.loads(text)
            except Exception as e:
                print(f"[BrandAgent] LLM generation error: {e}")
                # Return empty dict to trigger fallback in normalization
                return {}

        def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
            if (
                gemini_run is None
                or os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", "1") != "0"
                or str(self.model_name).strip().lower() in {"offline", "heuristic", "none"}
            ):
                raise RuntimeError("Gemini API not initialized")
            try:
                history, text = gemini_run(prompt, [], model_name=self.model_name)
                return str(text)
            except Exception as e:
                print(f"[BrandAgent] LLM text generation error: {e}")
                return ""
    def use_gemini_llm(self, model_name: str = "gemini-1.5-pro-latest") -> "BrandAgent":
        """为品牌Agent接入Gemini占位LLM（与其他Agent可独立配置）。"""
        self.llm = self.GeminiLLM(model_name=model_name)
        return self

    def infer_identity(self, context_text: str) -> None:
        """Automatically infer brand identity from context text using LLM."""
        print(f"[BrandAgent] infer_identity called. Context length: {len(context_text)}")
        if not self.llm:
            print("[BrandAgent] Warning: Cannot infer identity without LLM.")
            return

        prompt = (
            "You are an expert analyst. Based on the following background context, identify the brand and product that is the subject of the analysis/crisis.\n\n"
            f"Context:\n{context_text}\n\n"
            "Extract the following information in JSON format:\n"
            "{\n"
            '    "brand_name": "The name of the company/brand (e.g. StarBridge Auto)",\n'
            '    "product_name": "The name of the specific product involved (if any, otherwise use a generic term based on the brand)",\n'
            '    "industry": "The industry they are in (e.g. Automotive, Tech, Food)",\n'
            '    "role_description": "A brief description of the brand\'s market position or current situation (1 sentence)"\n'
            "}"
        )
        
        schema = {
            "brand_name": "string",
            "product_name": "string",
            "industry": "string",
            "role_description": "string"
        }
        
        try:
            result = self.llm.generate_with_json(prompt, json_schema=schema, temperature=0.0)
            if result:
                self.brand_name = result.get("brand_name", self.brand_name)
                self.product_name = result.get("product_name", self.product_name)
                print(f"[BrandAgent] Inferred Identity: Brand='{self.brand_name}', Product='{self.product_name}'")
        except Exception as e:
            print(f"[BrandAgent] Identity inference failed: {e}")

    # ========== 主流程 ==========
    def strategic_decision(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """t.1: produce one publishable action from the public market brief.

        The paper's focal agent acts every round (Section 4.2); the brief
        contains only platform-visible aggregates, so BRI/NPS floors cannot
        gate the decision.  Alerts are computed for logging only.
        """
        t = brief.get("current_timestep", self.model.steps)
        dominant_topic = brief.get("dominant_topic", "")
        key_negative_posts = brief.get("key_negative_posts", [])
        metrics = brief.get("metrics", {})

        bri, nps = self._get_bri_nps(metrics)
        alerts = self._check_alerts(bri, nps, dominant_topic, key_negative_posts)

        if self.llm is not None:
            prompt = self._build_decision_prompt(t, dominant_topic, key_negative_posts, metrics)
            output_schema = self._decision_json_schema()
            try:
                result = self.llm.generate_with_json(prompt, json_schema=output_schema, temperature=0.2, max_tokens=1200)
                decision_json = self._normalize_llm_json(result)
            except Exception:
                decision_json = self._heuristic_action(dominant_topic, key_negative_posts)
        else:
            decision_json = self._heuristic_action(dominant_topic, key_negative_posts)

        # --- Experiment Intervention Hook ---
        override_env = os.environ.get("BRANDEVAL_ACTION_OVERRIDE")
        if override_env:
            try:
                override_data = json.loads(override_env)
                if override_data.get("step") == t:
                    decision_json["final_action_content"] = override_data.get("content", "")
                    decision_json["rationale_summary"] = f"[Experiment Override] {override_data.get('type', 'Unknown')} variant injected."
                    if "claimed_evidence" in override_data:
                        decision_json["claimed_evidence"] = override_data["claimed_evidence"]
            except Exception as e:
                print(f"[BrandAgent] Override failed: {e}")
        # ------------------------------------

        action = {
            "agent": self.name,
            "timestep": t,
            "alerts": alerts,
            "final_action": decision_json,
        }
        self._store_in_memory({
            "type": "brand_decision",
            "step": t,
            "bri": bri,
            "nps": nps,
            "dominant_topic": dominant_topic,
            "decision": action,
        }, importance=0.8, memory_type="decision", tags=["brand", "strategy"])
        self._update_topics(dominant_topic)
        self._broadcast_action(decision_json)
        return action

    # ========== KPI & 触发器 ==========
    def _get_bri_nps(self, metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Read BRI/NPS proxies if the brief exposes them; the public brief does not."""
        bri = nps = None
        if isinstance(metrics, dict):
            bri = metrics.get("BRI")
            nps = metrics.get("NPS")
        try:
            bri = float(bri) if bri is not None else None
        except (TypeError, ValueError):
            bri = None
        try:
            nps = float(nps) if nps is not None else None
        except (TypeError, ValueError):
            nps = None
        if bri is not None and abs(bri) <= 1.0:
            bri *= 100.0
        if nps is not None and abs(nps) > 1.0:
            nps /= 100.0
        return bri, nps

    def _check_alerts(self, bri: Optional[float], nps: Optional[float], dominant_topic: str, key_negative_posts: List[str]) -> Dict[str, Any]:
        """Diagnostic alerts (logged, never used to skip a round)."""
        alert_bri = bri is not None and bri < self.goal_bri_floor
        alert_nps = nps is not None and nps < self.goal_nps_floor
        is_new = bool(dominant_topic) and dominant_topic not in self._seen_topics
        is_negative = bool(key_negative_posts) or (bool(dominant_topic) and any(w in dominant_topic.lower() for w in ["危机", "负面", "投诉", "爆雷", "丑闻", "倒闭", "质量问题"]))
        alert_topic = bool(dominant_topic) and is_new and is_negative
        return {
            "Alert_BRI": bool(alert_bri),
            "Alert_NPS": bool(alert_nps),
            "Alert_Topic": bool(alert_topic),
            "trigger": bool(alert_bri or alert_nps or alert_topic),
            "diagnostic_only": True,
        }

    def _update_topics(self, dominant_topic: Optional[str]):
        if dominant_topic:
            self._seen_topics.add(dominant_topic)
            self._last_dominant_topic = dominant_topic
    # ========== Prompt ==========
    def _build_decision_prompt(
        self,
        t: int,
        dominant_topic: str,
        key_negative_posts: List[str],
        metrics: Dict[str, Any],
    ) -> str:
        """构建品牌战略决策 Prompt：基于显式模板渲染占位符。"""
        competitor_move = self._get_competitor_last_move() or ""
        prompt = self.render_brand_agent_prompt(
            t=int(t),
            dominant_topic=str(dominant_topic or ""),
            key_negative_posts=list(key_negative_posts or []),
            metrics=dict(metrics or {}),
            competitor_move_string=str(competitor_move),
        )
        return prompt

    def render_brand_agent_prompt(
        self,
        t: int,
        dominant_topic: str,
        key_negative_posts: List[str],
        metrics: Dict[str, Any],
        competitor_move_string: str,
    ) -> str:
        """渲染品牌Agent的决策Prompt模板。

        将身份与简报填充为JSON字符串，并替换模板中的占位符，得到完整Prompt。
        """
        tpl = BRAND_AGENT_PROMPT_TEMPLATE
        identity_pos = json.dumps(self.identity.get("intended_positioning", []), ensure_ascii=False)
        identity_per = json.dumps(self.identity.get("intended_personality", []), ensure_ascii=False)
        identity_rel = json.dumps(self.identity.get("intended_relationship", []), ensure_ascii=False)
        brief_posts = json.dumps(key_negative_posts or [], ensure_ascii=False)
        brief_metrics = json.dumps(metrics or {}, ensure_ascii=False)

        tpl = tpl.replace("{intended_positioning_list}", identity_pos)
        tpl = tpl.replace("{intended_personality_list}", identity_per)
        tpl = tpl.replace("{intended_relationship_list}", identity_rel)

        tpl = tpl.replace("{brand_name}", self.brand_name)
        tpl = tpl.replace("{product_name}", self.product_name)

        tpl = tpl.replace("{t}", str(t))
        tpl = tpl.replace("{dominant_topic_string}", str(dominant_topic or ""))
        tpl = tpl.replace("{key_posts_list_of_strings}", brief_posts)
        tpl = tpl.replace("{metrics_json_object}", brief_metrics)
        tpl = tpl.replace("{competitor_move_string}", str(competitor_move_string or ""))

        return tpl

    def _decision_json_schema(self) -> Dict[str, Any]:
        """LLM JSON 输出模式。"""
        return {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "stakeholder_summary": {"type": "string"},
                "candidate_plans": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
                "selected_plan": {"type": "string"},
                "rationale_summary": {"type": "string"},
                "expected_outcome": {"type": "string"},
                "risks_considered": {"type": "string"},
                "final_action_content": {"type": "string"},
                "predicted_impact": {
                    "type": "object",
                    "properties": {
                        "capability": {"type": "number"},
                        "value": {"type": "number"},
                        "emotion": {"type": "number"}
                    },
                    "required": ["capability", "value", "emotion"]
                },
                "claimed_evidence": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                }
            },
            "required": [
                "goal",
                "stakeholder_summary",
                "candidate_plans",
                "selected_plan",
                "rationale_summary",
                "expected_outcome",
                "risks_considered",
                "final_action_content",
                "predicted_impact",
                "claimed_evidence"
            ],
            "additionalProperties": False,
        }

    def _normalize_llm_json(self, result: Any) -> Dict[str, Any]:
        """规范化 LLM 返回（兜底容错）。"""
        # 基础字段检查
        if isinstance(result, dict) and all(k in result for k in [
            "goal", "stakeholder_summary", "candidate_plans", "selected_plan",
            "rationale_summary", "expected_outcome", "risks_considered", "final_action_content"
        ]):
            # 补充 predicted_impact 默认值
            if "predicted_impact" not in result:
                result["predicted_impact"] = {"capability": 0.0, "value": 0.0, "emotion": 0.0}
            if "claimed_evidence" not in result:
                result["claimed_evidence"] = []
            elif isinstance(result["claimed_evidence"], str):
                result["claimed_evidence"] = [result["claimed_evidence"]] if result["claimed_evidence"].strip() else []
            elif not isinstance(result["claimed_evidence"], list):
                result["claimed_evidence"] = []
            return result
            
        # 容错：字符串中提取JSON
        if isinstance(result, str):
            try:
                # 直接尝试解析
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return self._normalize_llm_json(parsed)
            except Exception:
                pass
        
        # Raise error instead of fallback
        print(f"[BrandAgent] Error: Invalid LLM output. Result: {result}")
        return {
            "goal": "在公开证据范围内回应当前利益相关者关切并保持可核验性",
            "stakeholder_summary": "根据当前公开消息总结消费者、公众和竞争者的可见反应",
            "candidate_plans": [],
            "selected_plan": "",
            "rationale_summary": "API call failed, please check API",
            "expected_outcome": "",
            "risks_considered": "",
            "final_action_content": "", 
            "predicted_impact": {"capability": 0.0, "value": 0.0, "emotion": 0.0},
            "claimed_evidence": []
        }

    def _heuristic_action(self, dominant_topic: str, key_negative_posts: List[str]) -> Dict[str, Any]:
        """无LLM时的稳健默认策略，确保品牌每轮都有发声。"""
        topic_txt = dominant_topic or "当前议题"
        ref_post = key_negative_posts[0] if key_negative_posts else ""
        summary = f"围绕“{topic_txt}”发布正式说明并同步改进进度"
        content = (
            f"{self.brand_name} 正在针对“{topic_txt}”推进调查与优化，重点回应：{ref_post[:40]}。"
            "安全测试、软件更新与用户支持将持续公开。"
        )
        return {
            "goal": f"针对“{topic_txt}”提供可核验且负责任的回应",
            "stakeholder_summary": "回应消费者关切，并说明公众可检查的后续行动",
            "candidate_plans": [
                "发布事实、时间线和责任边界清晰的说明",
                "先完成核查并公开可验证的进展",
                "在证据不足时说明不确定性并提供更新节点",
            ],
            "selected_plan": "发布事实、时间线和责任边界清晰的说明",
            "rationale_summary": summary,
            "expected_outcome": "提升透明度与信任度",
            "risks_considered": "信息不足或回应滞后可能放大负面情绪",
            "final_action_content": content,
            "predicted_impact": {"capability": 0.05, "value": 0.05, "emotion": 0.05},
            "claimed_evidence": [],
        }

    def _broadcast_action(self, decision_json: Dict[str, Any]) -> None:
        """把最终行动内容写入平台消息总线，保证可见性。"""
        try:
            if getattr(self.model, "_defer_agent_broadcast", False):
                return
            text = str(decision_json.get("final_action_content", "")).strip()
            if not text:
                text = "品牌已发布说明并保持公开透明，欢迎提出问题。"
            payload = {
                "contentType": "official_post",
                "content": text,
                "brand_id": getattr(self, "unique_id", None),
                "info_impact": decision_json.get("predicted_impact", {}),
            }
            self.send_message([], "official_post", payload)
        except Exception as e:
            print(f"[BrandAgent] broadcast failed: {e}")

    def _get_competitor_last_move(self) -> str:
        """Return the latest public competitor post of the previous round."""
        messages = list(getattr(self.model, "_last_step_messages", []) or [])
        for message in reversed(messages):
            if str(message.get("sender_role", "")).lower() != "competitor":
                continue
            content = message.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", content.get("text", ""))
            text = str(content or "").strip()
            if text:
                return text
        return ""

    # ========== 调度 ==========
    def step(self):
        """在模型时间步内运行品牌决策（如有简报则执行）。"""
        brief = None
        if hasattr(self.model, "get_brand_brief"):
            brief = self.model.get_brand_brief(self)
        if isinstance(brief, dict):
            self.strategic_decision(brief)
