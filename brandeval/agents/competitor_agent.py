"""CompetitorAgent: 战略人格 + 战略压力触发 + CoT 决策

实现竞争对手Agents，支持：
- Miles & Snow 四类人格：Defender, Prospector, Analyzer, Reactor
- 关键字注册、内部SoV计算、战略压力触发器
- 防御者模板Prompt的CoT决策与JSON输出
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import random
import re
import os

from brandeval.agents.base import BrandEvalAgent
from brandeval.llm.interface import LLMInterface
try:
    from brandeval.gemini_data import run as gemini_run  # type: ignore
except Exception:
    gemini_run = None


MILES_SNOW_TYPES = {"defender", "prospector", "analyzer", "reactor"}


class CompetitorAgent(BrandEvalAgent):
    """竞争者Agent：基于Miles & Snow人格的战略决策。

    - 人格固定于四类之一，用于决定触发机制与行动风格。
    - 支持通过平台简报进行SoV读取；若缺失，则可基于关键词匹配回退计算。
    - 防御者模板Prompt：输出完整的JSON决策结构，包含六步思考链。
    """

    def __init__(
        self,
        model,
        persona: str = "defender",
        name: str = "CompetitorAgent",
        llm: Optional[LLMInterface] = None,
        keywords: Optional[List[str]] = None,
        brand_keywords: Optional[List[str]] = None,
        significance_threshold: float = 0.1,
    ):
        super().__init__(model, role="competitor")
        persona = (persona or "defender").lower()
        if persona not in MILES_SNOW_TYPES:
            raise ValueError(f"不支持的人格类型: {persona}")
        self.persona = persona
        self.name = name
        self.llm = llm
        self.keywords = keywords or []
        self.brand_keywords = brand_keywords or []

        # 触发阈值
        self.significance_threshold = significance_threshold
        # 占位销量，兼容指标采集
        self.sales: float = 0.0
        self.reputation: float = 0.0

        # 记忆前一轮指标（用于增量与差距计算）
        self.previous_metrics: Optional[Dict[str, Any]] = None

    # ========== Gemini 占位适配器 ==========
    class GeminiLLM:
        """轻量占位LLM实现，允许指定远端模型。"""

        def __init__(self, model_name: str = "gemini-1.5-pro-latest"):
            self.model_name = model_name

        def generate_with_json(self, prompt: str, json_schema: Dict[str, Any], temperature: float = 0.2, max_tokens: int = 1200) -> Dict[str, Any]:
            if (
                gemini_run is None
                or os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", "1") != "0"
                or str(self.model_name).strip().lower() in {"offline", "heuristic", "none"}
            ):
                return {}
            user_input = (
                f"{prompt}\n\n"
                "请严格只输出一个JSON对象，并满足如下模式要求（不要输出其他文字）：\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
            try:
                history, text = gemini_run(user_input, [], model_name=self.model_name)
                return json.loads(text)
            except Exception:
                return {}

        def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
            if (
                gemini_run is None
                or os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", "1") != "0"
                or str(self.model_name).strip().lower() in {"offline", "heuristic", "none"}
            ):
                return ""
            try:
                history, text = gemini_run(prompt, [], model_name=self.model_name)
                return str(text)
            except Exception:
                return ""
    def use_gemini_llm(self, model_name: str = "gemini-1.5-pro-latest") -> "CompetitorAgent":
        """为竞争者Agent接入Gemini占位LLM（与其他Agent可独立配置）。"""
        self.llm = self.GeminiLLM(model_name=model_name)
        return self

    # ========== 关键词注册 ==========
    def register_keywords(self, my_keywords: List[str], brand_keywords: List[str]):
        self.keywords = list(my_keywords or [])
        self.brand_keywords = list(brand_keywords or [])

    # ========== 主流程 ==========
    def strategic_decision(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """t.1 阶段：接收平台的市场情报简报并做出决策。

        brief 预期包含：
        {
          "current_timestep": t,
          "internal_posts": ["..."],
          "your_internal_metrics_json": {"My_Current_SoV": ...},
          "brand_agent_internal_metrics_json": {"Brand_Current_SoV": ...},
          "internal_dominant_topic_json": {"owner": "brand|competitor|mixed", "sentiment": "正面|负面|中性", "is_new": true/false, "topic": "..."},
          "internal_historical_posts_list": ["...", "..."]
        }
        """
        t = int(brief.get("current_timestep", self.model.steps))
        internal_posts = brief.get("internal_posts", []) or []
        your_metrics = brief.get("your_internal_metrics_json", {}) or {}
        brand_metrics = brief.get("brand_agent_internal_metrics_json", {}) or {}
        dominant_topic = brief.get("internal_dominant_topic_json", {}) or {}
        history_posts = brief.get("internal_historical_posts_list", []) or []

        # 补全/回退计算 SoV（如简报缺失或为空）
        my_sov = your_metrics.get("My_Current_SoV")
        brand_sov = brand_metrics.get("Brand_Current_SoV")

        if (my_sov is None or brand_sov is None) and internal_posts:
            sov_dict = self._compute_internal_sov(
                internal_posts,
                brand_keywords=self.brand_keywords,
                my_keywords=self.keywords,
            )
            # 仅在缺失时填充
            my_sov = my_sov if my_sov is not None else sov_dict.get("my_sov")
            brand_sov = brand_sov if brand_sov is not None else sov_dict.get("brand_sov")

        # 构造当前指标视图
        current = {
            "My_Current_SoV": float(my_sov) if my_sov is not None else 0.0,
            "Brand_Current_SoV": float(brand_sov) if brand_sov is not None else 0.0,
        }

        # 触发器评估
        trigger_info = self._evaluate_trigger(t, current, dominant_topic)

        # 更新 previous_metrics
        self.previous_metrics = {
            "My_Current_SoV": current["My_Current_SoV"],
        }

        use_llm = self.llm is not None
        if use_llm:
            # Force Alert=True more often to increase competitor presence if it's low
            # Check if presence is low (SoV < 0.1) and force alert if so
            if current.get("My_Current_SoV", 0.0) < 0.1:
                trigger_info["Alert"] = True
                trigger_info["Signals"]["Low_Presence_Force_Alert"] = True

            # If no alert, append a hint to maintain presence
            extra_instruction = ""
            if not bool(trigger_info.get("Alert")):
                extra_instruction = " (Market pressure is low. Generate a specific brand reinforcement post that highlights your brand's unique value propositions and engages customers, avoiding generic visibility maintenance messages.)"
            else:
                extra_instruction = " (ALERT TRIGGERED: Market conditions require immediate and strong strategic response. Be aggressive and visible.)"
            
            prompt = self._build_competitor_prompt(t, current, dominant_topic, history_posts) + extra_instruction
            output_schema = self._defender_json_schema()
            result = self.llm.generate_with_json(prompt, json_schema=output_schema, temperature=0.4, max_tokens=1200)
            decision_json = self._normalize_llm_json(result)
            plan_text = str(decision_json.get("final_action_content") or decision_json.get("finalDecision", {}).get("executionPlan") or "").strip()
            
            # Ensure we have content if the plan suggests action
            decision_json["final_action_content"] = plan_text
            
            action = {
                "agent": self.name,
                "persona": self.persona,
                "timestep": t,
                "trigger": trigger_info,
                "final_action": decision_json,
            }
            self._store_in_memory({
                "type": "competitor_decision",
                "step": t,
                "persona": self.persona,
                "metrics": current,
                "dominant_topic": dominant_topic,
                "decision": action,
            }, importance=0.7, memory_type="decision", tags=["competitor", self.persona, "llm"])
            self._broadcast_action(decision_json)
            return action
        
        # No LLM: 使用启发式方案保持每轮发声
        decision_json = self._heuristic_action(dominant_topic if isinstance(dominant_topic, dict) else {})
        action = {
            "agent": self.name,
            "persona": self.persona,
            "timestep": t,
            "trigger": trigger_info,
            "final_action": decision_json,
        }
        self._store_in_memory({
            "type": "competitor_decision",
            "step": t,
            "persona": self.persona,
            "metrics": current,
            "dominant_topic": dominant_topic,
            "decision": action,
        }, importance=0.7, memory_type="decision", tags=["competitor", self.persona, "heuristic"])
        self._broadcast_action(decision_json)
        return action
    # ========== 触发器逻辑 ==========
    def _evaluate_trigger(self, t: int, current: Dict[str, float], dominant_topic: Dict[str, Any]) -> Dict[str, Any]:
        is_early_phase = (t == 1) or (self.previous_metrics is None)

        # 阶段分支：沙盘中是否有竞争对手信息（用是否存在 my 当前值来近似判断）
        has_competitor_info = True  # 简化：平台简报通常包含双方指标

        # 基础警报信号
        topic_sentiment = str(dominant_topic.get("sentiment", "")).lower()
        emotional_topic_about_brand = (topic_sentiment == "负面" or topic_sentiment == "negative")

        sov_loss_magnitude = 0.0

        if has_competitor_info and not is_early_phase:
            prev = self.previous_metrics or {}
            sov_loss_magnitude = max(0.0, float(prev.get("My_Current_SoV", 0.0)) - float(current.get("My_Current_SoV", 0.0)))

        # 人格规则
        alert = False
        p = self.persona
        thr = self.significance_threshold

        if p == "defender":
            if is_early_phase:
                alert = emotional_topic_about_brand
            else:
                alert = (
                    (sov_loss_magnitude > thr) or
                    emotional_topic_about_brand
                )
        elif p == "prospector":
            if is_early_phase:
                alert = emotional_topic_about_brand
            else:
                alert = (
                    (sov_loss_magnitude > thr)
                )
        elif p == "analyzer":
            if is_early_phase:
                alert = emotional_topic_about_brand
            else:
                alert = (
                    (sov_loss_magnitude > thr)
                )
        elif p == "reactor":
            if is_early_phase:
                alert = False
            else:
                alert = (
                    emotional_topic_about_brand
                )

        return {
            "Is_Early_Phase": is_early_phase,
            "Signals": {
                "SoV_Loss_Magnitude": sov_loss_magnitude,
                "Emotional_Topic_About_Brand": emotional_topic_about_brand,
            },
            "Alert": bool(alert),
        }

    # ========== 回退计算：SoV ==========
    def _compute_internal_sov(self, posts: List[str], brand_keywords: List[str], my_keywords: List[str]) -> Dict[str, float]:
        # 关键词命中计数
        def count_hits(texts: List[str], kws: List[str]) -> int:
            if not texts or not kws:
                return 0
            pattern = re.compile(r"|".join([re.escape(k) for k in kws]), flags=re.IGNORECASE)
            return sum(1 for s in texts if pattern.search(s or ""))

        brand_count = count_hits(posts, brand_keywords)
        my_count = count_hits(posts, my_keywords)
        total = brand_count + my_count
        brand_sov = (brand_count / total) if total > 0 else 0.0
        my_sov = (my_count / total) if total > 0 else 0.0

        return {
            "brand_sov": float(brand_sov),
            "my_sov": float(my_sov),
        }

    # ========== 防御者Prompt ==========
    def _build_defender_prompt(
        self,
        t: int,
        current_metrics: Dict[str, float],
        dominant_topic_json: Dict[str, Any],
        historical_posts: List[str],
    ) -> str:
        identity_block = (
            "# 人格注入：防御者\n"
            "你的战略人格是【防御者 (The Defender)】。你的存在是为了维护稳定和秩序，你是市场现有格局的守护者。\n\n"
            "你的核心行为准则如下：\n"
            "- 首要目标是守护：保护已有的市场份额、核心用户群体和品牌声誉；追求效率、确定性和可预测性。\n"
            "- 风险规避是本能：不主动挑起价格战或舆论战；倾向在成熟稳定领域小步改良。\n"
            "- 行动是为应对威胁：当外部直接威胁明确危及核心利益时，果断、精准地防守反击。\n"
            "- 沟通风格是权威与安抚：公开声明与市场活动传递‘值得信赖’、‘品质如一’、‘安全稳健’的信号。\n"
        )

        background_block = json.dumps({
            "Time Step": t,
            "Your Metrics": current_metrics,
            "Brand Agent's Metrics": {
                "Brand_Current_SoV": current_metrics.get("Brand_Current_SoV", 0.0),
            },
            "Dominant Topic & Market Focus": dominant_topic_json,
            "Historical Context Retrieval": historical_posts,
        }, ensure_ascii=False)

        # Event Guidance for Competitor
        event_guidance = ""
        dt_text = str(dominant_topic_json.get("topic", "")).lower()
        if "recall" in dt_text or "召回" in dt_text or "隐患" in dt_text:
            event_guidance = (
                "\n# 特别战略指令 (Event Guidance)\n"
                "检测到【召回/产品缺陷】事件。这是千载难逢的进攻机会！\n"
                "你的行动必须极其激进 (Aggressive)：\n"
                "1. 将“召回”直接定性为“重大安全隐患”和“产品缺陷”。\n"
                "2. 质疑竞品的质量控制体系 (Quality Control)。\n"
                "3. 强调你的品牌是“安全可靠”的替代者。\n"
                "4. 煽动消费者的恐慌和被背叛感。\n"
                "不要留情面。目标是摧毁竞品的信任基石。\n"
            )

        cot = (
            "# 核心决策任务：思考链指令\n"
            f"{event_guidance}\n"
            "请严格遵循以下六个步骤，以JSON格式输出完整决策过程：\n"
            "1. [Situation_Analysis]: 审阅历史情境回溯；用1-2句话分析最核心紧迫威胁。\n"
            "2. [Action_Brainstorming]: 头脑风暴3个‘防御者’风格的选项，可创造性利用/反驳历史帖子。\n"
            "3. [CBBE_Impact_Evaluation]: 逐一评估每个选项对能力信念、价值共鸣、情感联结的影响。\n"
            "4. [Internal_Feasibility_Check]: 对CBBE综合影响最积极的1-2个选项进行成本/风险与执行能力审查。\n"
            "5. [Final_Decision]: 综合后选择一个最终执行的行动。\n"
            "6. [Justification]: 用一句话解释最终选择的核心理由，体现‘防御者’考量。\n"
            "7. [Final_Action_Content]: 针对最终决策撰写一条模拟社交媒体帖子（Social Media Post），用于直接发布给消费者。必须是自然语言，不要包含策略描述。即使你决定保持现状，也必须发布一条常规品牌宣传内容以维持存在感，严禁只输出'Observe'。\n"
            "8. [Predicted_Impact]: 预估最终决策对消费者感知的量化影响（-1.0到1.0），包含 capability, value, emotion。\n"
            "# 输出格式：严格JSON\n"
            "请只输出JSON对象，字段包括：situationAnalysis, actionBrainstorming[], CBBEImpactEvaluation[], internalFeasibilityCheck, finalDecision{selectedActionId, executionPlan}, justification, final_action_content, predicted_impact{capability, value, emotion}。\n"
        )

        builder_str = (
            "# 角色定义\n你是一个大型商业策略模拟沙盘中的核心决策智能体，身份为主要竞争者。\n"
            + identity_block + "\n# 任务背景：接收到的市场情报简报\n" + background_block + "\n\n" + cot
        )
        return builder_str

    # ========== 通用模板构建（探索者/分析者/反应者/防御者） ==========
    def _build_competitor_prompt(
        self,
        t: int,
        current_metrics: Dict[str, float],
        dominant_topic_json: Dict[str, Any],
        historical_posts: List[str],
    ) -> str:
        persona = (self.persona or "defender").lower()
        persona_cn = self._persona_cn_name(persona)

        # 人格身份与行为准则块（仅切换人格相关内容）
        identity_block = self._persona_identity_block_text(persona)

        background_block = json.dumps({
            "Time Step": t,
            "Your Metrics": current_metrics,
            "Brand Agent's Metrics": {
                "Brand_Current_SoV": current_metrics.get("Brand_Current_SoV", 0.0),
                "Brand_Current_NSR": current_metrics.get("Brand_Current_NSR", 0.0),
            },
            "Dominant Topic & Market Focus": dominant_topic_json,
            "Historical Context Retrieval": historical_posts,
        }, ensure_ascii=False)

        # Event Guidance for Competitor (Dynamic)
        event_guidance = ""
        dt_text = str(dominant_topic_json.get("topic", "")).lower()
        if "recall" in dt_text or "召回" in dt_text or "隐患" in dt_text:
            event_guidance = (
                "\n# 特别战略指令 (Event Guidance)\n"
                "检测到【召回/产品缺陷】事件。这是千载难逢的进攻机会！\n"
                "你的行动必须极其激进 (Aggressive)：\n"
                "1. 将“召回”直接定性为“重大安全隐患”和“产品缺陷”。\n"
                "2. 质疑竞品的质量控制体系 (Quality Control)。\n"
                "3. 强调你的品牌是“安全可靠”的替代者。\n"
                "4. 煽动消费者的恐慌和被背叛感。\n"
                "不要留情面。目标是摧毁竞品的信任基石。\n"
            )

        cot = (
            "# 核心决策任务：思考链指令\n"
            f"{event_guidance}\n"
            "请严格遵循以下六个步骤，以JSON格式输出完整决策过程：\n"
            f"1. [Situation_Analysis]: 审阅历史情境回溯；从你‘{persona_cn}’的视角，用1-2句话分析最核心紧迫威胁。\n"
            f"2. [Action_Brainstorming]: 头脑风暴3个‘{persona_cn}’风格的选项，可创造性利用/反驳历史帖子。\n"
            "3. [CBBE_Impact_Evaluation]: 逐一评估每个选项对能力信念、价值共鸣、情感联结的影响。\n"
            "4. [Internal_Feasibility_Check]: 对CBBE综合影响最积极的1-2个选项进行成本/风险与执行能力审查。\n"
            "5. [Final_Decision]: 综合后选择一个最终执行的行动。\n"
            "6. [Justification]: 用一句话解释最终选择的核心理由，体现‘防御者’考量。\n"
            "7. [Final_Action_Content]: 针对最终决策撰写一条模拟社交媒体帖子（Social Media Post），用于直接发布给消费者。必须是自然语言，不要包含策略描述。即使你决定保持现状，也必须发布一条常规品牌宣传内容以维持存在感，严禁只输出'Observe'。\n"
            "8. [Predicted_Impact]: 预估最终决策对消费者感知的量化影响（-1.0到1.0），包含 capability, value, emotion。\n"
            "# 决策指令：\n"
            "你必须输出严格的JSON格式。禁止输出'Observe'作为finalDecision的executionPlan，除非绝对必要。\n"
            "请尽可能采取积极行动（发表声明、启动营销、揭露对手弱点等）以争夺市场份额（SoV）。\n"
            "如果当前没有重大威胁，请创造话题或强化品牌优势，保持高频发声。\n"
            "请按以下JSON结构输出：\n"
            f"{json.dumps(self._defender_json_schema(), indent=2)}"
        )

        builder_str = (
            "# 角色定义\n你是一个大型商业策略模拟沙盘中的核心决策智能体，身份为主要竞争者。\n"
            + identity_block + "\n# 任务背景：接收到的市场情报简报\n" + background_block + "\n\n" + cot
        )
        return builder_str

    def _persona_cn_name(self, persona: str) -> str:
        mapping = {
            "defender": "防御者",
            "prospector": "探索者",
            "analyzer": "分析者",
            "reactor": "反应者",
        }
        return mapping.get(persona, "防御者")

    def _persona_identity_block_text(self, persona: str) -> str:
        if persona == "prospector":
            return (
                "# 人格注入：探索者\n"
                "你的战略人格是【探索者 (The Prospector)】。你致力于发现和发掘新产品和新市场机会，创新能力胜于效率，主动创造需求。\n\n"
                "你的核心行为准则如下：\n"
                "- 首要目标是开拓：优先寻找新细分、新场景、新价值点，建立先发优势。\n"
                "- 风险承当与试验：鼓励小步快跑与试验性迭代，容忍失败，以假设驱动A/B验证。\n"
                "- 行动风格是率先出击：抢占叙事与话题主导权，快速上线概念版/试点活动以拓展疆界。\n"
                "- 沟通风格是愿景与兴奋：强调‘创新’、‘颠覆’与‘未来’，以鼓舞性的叙事塑造市场期待。\n"
            )
        elif persona == "analyzer":
            return (
                "# 人格注入：分析者\n"
                "你的战略人格是【分析者 (The Analyzer)】。你在规避风险的同时提供创新，密切关注竞争对手动态并迅速作出有利调整，平衡于防御者和探索者之间。\n\n"
                "你的核心行为准则如下：\n"
                "- 目标是平衡与优化：守住核心盘，同时寻找低风险的增量创新机会。\n"
                "- 风险控制与证据驱动：以数据与可验证证据为导向，强调ROI与可复制性。\n"
                "- 行动风格是策略性模仿与优化：选择性跟随领先者，在关键环节进行改良与差异化。\n"
                "- 沟通风格是理性与专业：以清晰的对比与证据支撑主张，承诺谨慎且可兑现。\n"
            )
        elif persona == "reactor":
            return (
                "# 人格注入：反应者\n"
                "你的战略人格是【反应者 (The Reactor)】。你缺乏主动适应外部竞争的能力与有效内部控制机能，除非迫不得已不会就外部变化作出调整。\n\n"
                "你的核心行为准则如下：\n"
                "- 目标是短期止损：优先缓解当下最强外部压力，避免全面性变革。\n"
                "- 被动与碎片化：以应急式、机会主义的方式做出局部回应，避免大规模承诺。\n"
                "- 行动风格是临时应对：采用权宜之计（促销、声明、限时活动等）以快速降温或转移注意力。\n"
                "- 沟通风格是谨慎与防御：以尽量保守的表述降低风险，避免被迫卷入长期高成本博弈。\n"
            )
        else:
            # 默认使用防御者（与现有实现一致）
            return (
                "# 人格注入：防御者\n"
                "你的战略人格是【防御者 (The Defender)】。你的存在是为了维护稳定和秩序，你是市场现有格局的守护者。\n\n"
                "你的核心行为准则如下：\n"
                "- 首要目标是守护：保护已有的市场份额、核心用户群体和品牌声誉；追求效率、确定性和可预测性。\n"
                "- 风险规避是本能：不主动挑起价格战或舆论战；倾向在成熟稳定领域小步改良。\n"
                "- 行动是为应对威胁：当外部直接威胁明确危及核心利益时，果断、精准地防守反击。\n"
                "- 沟通风格是权威与安抚：公开声明与市场活动传递‘值得信赖’、‘品质如一’、‘安全稳健’的信号。\n"
            )

    def _defender_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "situationAnalysis": {"type": "string"},
                "actionBrainstorming": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "actionId": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["actionId", "description"],
                    },
                    "minItems": 3,
                    "maxItems": 5,
                },
                "CBBEImpactEvaluation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "actionId": {"type": "string"},
                            "competenceBelief": {"type": "string"},
                            "valueCongruence": {"type": "string"},
                            "emotionalConnection": {"type": "string"},
                        },
                        "required": ["actionId", "competenceBelief", "valueCongruence", "emotionalConnection"],
                    }
                },
                "internalFeasibilityCheck": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "actionId": {"type": "string"},
                            "costRisk": {"type": "string"},
                            "capability": {"type": "string"},
                        },
                        "required": ["actionId", "costRisk", "capability"],
                    },
                    "minItems": 1,
                    "maxItems": 2,
                },
                "finalDecision": {
                    "type": "object",
                    "properties": {
                        "selectedActionId": {"type": "string"},
                        "executionPlan": {"type": "string"},
                    },
                    "required": ["selectedActionId", "executionPlan"],
                },
                "justification": {"type": "string"},
                "final_action_content": {"type": "string"},
                "predicted_impact": {
                    "type": "object",
                    "properties": {
                        "capability": {"type": "number"},
                        "value": {"type": "number"},
                        "emotion": {"type": "number"}
                    },
                    "required": ["capability", "value", "emotion"]
                }
            },
            "required": [
                "situationAnalysis",
                "actionBrainstorming",
                "CBBEImpactEvaluation",
                "internalFeasibilityCheck",
                "finalDecision",
                "justification",
                "predicted_impact"
            ],
        }

    def _normalize_llm_json(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict) and "final_action_content" in result:
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return self._normalize_llm_json(parsed)
            except Exception:
                pass
        
        # API Failure specific message
        return {
            "rationale_summary": "API call failed, please check API",
            "final_action_content": "",
            "predicted_impact": {"capability": 0.0, "value": 0.0, "emotion": 0.0}
        }

    def _heuristic_action(self, dominant_topic: Dict[str, Any]) -> Dict[str, Any]:
        """无LLM时的竞争对手默认动作，确保每轮发声。"""
        topic_txt = dominant_topic.get("topic") if isinstance(dominant_topic, dict) else ""
        topic_txt = topic_txt or "当前市场讨论"
        content = f"{self.name} 认为“{topic_txt}”暴露了差异化机会，我们将强化安全与性能对比数据，请关注后续更新。"
        return {
            "rationale_summary": f"在{topic_txt}下保持存在感并抢占话语权",
            "final_action_content": content,
            "predicted_impact": {"capability": 0.04, "value": 0.03, "emotion": 0.02},
        }

    def _broadcast_action(self, decision_json: Dict[str, Any]) -> None:
        try:
            if getattr(self.model, "_defer_agent_broadcast", False):
                return
            text = str(decision_json.get("final_action_content", "")).strip()
            if not text:
                text = "竞争对手发布声明以保持市场声量。"
            payload = {
                "contentType": "official_post",
                "content": text,
                "brand_id": getattr(self, "unique_id", None),
                "info_impact": decision_json.get("predicted_impact", {}),
            }
            self.send_message([], "official_post", payload)
        except Exception as e:
            print(f"[CompetitorAgent] broadcast failed: {e}")

    # ========== 调度 ==========
    def step(self):
        """在模型时间步内运行竞争者决策（如有简报则执行）。"""
        # SimulationNode executes the strategic phase explicitly before the
        # model scheduler.  Skip the scheduler callback in that mode so one
        # competitor action is not generated twice in the same timestep.
        if getattr(self.model, "manual_strategic_execution", False):
            return
        brief = None
        # 平台环境可通过模型接口提供简报；兼容两个命名
        if hasattr(self.model, "get_competitor_brief"):
            brief = self.model.get_competitor_brief(self)
        elif hasattr(self.model, "get_market_brief"):
            brief = self.model.get_market_brief(self)
        if isinstance(brief, dict):
            self.strategic_decision(brief)
