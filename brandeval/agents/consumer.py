"""消费者智能体类
实现用户Agents的人格矩阵与TPB/CBBE动态状态更新：
- 人格：核心决策驱动力(5) × 初始立场(3) 组合为15种原型
- 感知状态(CBBE)：能力信念、价值共鸣、情感联结 ∈ [-1,1]
- 意图状态(TPB)：购买意向 ∈ [0,1]，宣传意向 ∈ [-1,1]
- 级联更新：ELM β → EMA 记忆衰减 → TPB w → 归一化 → 立场派生
"""

from typing import Dict, List, Any, Optional, Tuple
import hashlib
import random
import numpy as np
import json
import os

from brandeval.agents.base import BrandEvalAgent
from brandeval.config.loaders import DEFAULT_BETA, DEFAULT_W
try:
    from brandeval.gemini_data import run as gemini_run, DEFAULT_MODEL  # type: ignore
except Exception:
    gemini_run = None
    DEFAULT_MODEL = "gemini-2.5-flash-lite"


CORE_DRIVERS = [
    "value_oriented",        # 价值导向
    "performance_oriented",  # 性能导向
    "experience_oriented",   # 体验导向
    "social_identity_oriented",  # 社会认同导向
    "practical_safety_oriented"   # 实用/安全导向
]

STANCES = [
    "supporter",  # 支持者
    "neutral",    # 中立者
    "opponent"    # 反对者
]

# 15种原型标签（示意映射）
PROTOTYPE_LABELS = {
    ("value_oriented", "supporter"): "竞品性价比卫士",
    ("value_oriented", "neutral"): "理性比价党",
    ("value_oriented", "opponent"): "米粉性价比吹",
    ("performance_oriented", "supporter"): "技术铁粉",
    ("performance_oriented", "neutral"): "硬核参数对比党",
    ("performance_oriented", "opponent"): "技术吹毛求疵者",
    ("experience_oriented", "supporter"): "设计美学粉",
    ("experience_oriented", "neutral"): "颜值种草党",
    ("experience_oriented", "opponent"): "品牌调性批评家",
    ("social_identity_oriented", "supporter"): "销量冠军粉",
    ("social_identity_oriented", "neutral"): "口碑跟随者",
    ("social_identity_oriented", "opponent"): "唱衰销量者",
    ("practical_safety_oriented", "supporter"): "家庭首选推荐官",
    ("practical_safety_oriented", "neutral"): "全家出行考察员",
    ("practical_safety_oriented", "opponent"): "安全隐患放大器",
}

# 用户Agent社交媒体生成提示模板（显式呈现）
CONSUMER_SOCIAL_PROMPT_TEMPLATE = (
    """
Role: Social Media Consumer
Persona:
* Core Driver: {core_driver}
* Initial Stance: {initial_stance}
* Example Behavior: "{persona_example}"

Internal State (t):
* Competence: {competenceBelief_score}
* Value: {valueCongruence_score}
* Connection: {emotionalConnection_score}
* Advocacy: {advocacyIntent_score}
* Stance: {current_stance}

Context:
* Action Type: {contentType}
* Reply To: "{inReplyTo_post_summary}"
* Topic: "{dominant_topic}"

{event_guidance}

Task:
Generate a short, realistic social media post or comment based on your persona and state.

Requirements:
1. Tone must match your [Stance].
2. Intensity must match [Advocacy] (higher = more intense).
3. Focus on [Core Driver] topics.
4. Must be relevant to [Reply To] or [Topic].
5. Natural, casual language (Chinese).
6. IF [Stance] is opponent, be critical/sarcastic. IF neutral, be hesitant. IF supporter, be defensive but acknowledge issues if evident.

OUTPUT FORMAT:
Output ONLY the text of the post/comment. Do NOT include any headers, explanations, or labels like "Content:".
Just the text.
"""
).strip()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Consumer(BrandEvalAgent):
    """消费者智能体
    
    代表市场中的消费者，具有品牌认知、情绪和购买意向等属性。
    消费者会接收来自品牌、竞品和其他消费者的信息，并据此更新自己的状态。
    """
    
    def __init__(self, model, influence_susceptibility=None, loyalty=None,
                 core_driver: Optional[str] = None, initial_stance: Optional[str] = None,
                 learning_rate_lambda: Optional[float] = None,
                 memory_volatility: Optional[float] = None,
                 persona_seed: Optional[int] = None,
                 persona_seed_offset: Optional[int] = None,
                 persona_label: Optional[str] = None):
        """初始化消费者智能体
        
        Args:
            model: 模型实例
            influence_susceptibility: 受影响程度，范围[0, 1]
            loyalty: 品牌忠诚度，范围[0, 1]
        """
        super().__init__(model, role="consumer")
        
        # s_u ~ U(0,1): information-retention susceptibility used by Eq. (10).
        self.influence_susceptibility = influence_susceptibility if influence_susceptibility is not None else self.random.random()
        self.loyalty = loyalty if loyalty is not None else self.random.random()

        # 人格属性
        self.core_driver = core_driver if core_driver in CORE_DRIVERS else self.random.choice(CORE_DRIVERS)
        self.current_stance_global = initial_stance if initial_stance in STANCES else self.random.choice(STANCES)
        self.prototype_label = PROTOTYPE_LABELS.get((self.core_driver, self.current_stance_global), "generic_consumer")
        hp = getattr(model, "hyperparams", {}) or {}
        # Base update rate λ_u of Eq. (1); the effective rate also multiplies the
        # global decay factor and the driver-specific group factor (see _update_rate).
        self.learning_rate_lambda = float(learning_rate_lambda) if learning_rate_lambda is not None else float(hp.get("consumer_update_rate", 0.15))
        # Eq. (1) is noise-free; volatility is an explicit opt-in extension (default 0).
        self.memory_volatility = float(memory_volatility) if memory_volatility is not None else float(hp.get("memory_volatility", 0.0))
        self.persona_seed = persona_seed
        self.persona_seed_offset = persona_seed_offset
        self.persona_label = persona_label
        # ``start_benchmark`` already creates one focal UGC post while
        # processing the benchmark message.  The following all-brands hook is
        # suppressed once to prevent a second t=0 batch.
        self._skip_next_all_brands_generation = False
        
        # 消费者对各品牌的认知和态度
        self.brand_awareness = {}  # 品牌知名度 [0,1]
        self.brand_sentiment = {}  # 对品牌的情绪 [-1,1]
        self.purchase_intention = {}  # 购买意向 [0,1]（兼容旧字段）

        # CBBE感知状态与TPB意图（按品牌维度）
        self.cbbe_state: Dict[str, Dict[str, float]] = {}  # {brand_id: {capability_belief, value_resonance, emotional_bond}}
        self.intent_state: Dict[str, Dict[str, float]] = {}  # {brand_id: {purchase_intent, advocacy_intent}}
        self.stance_by_brand: Dict[str, str] = {}  # {brand_id: stance}
        
        # 初始化对所有品牌的认知
        for brand in model.agents_by_type.get("brand", []):
            bid = brand.unique_id
            self.register_brand(bid)

    def register_brand(self, brand_id: str):
        """Register a brand and realize the persona's initial state toward it."""
        if brand_id in self.brand_awareness:
            return
        self.brand_awareness[brand_id] = 0.0
        self.brand_sentiment[brand_id] = 0.0
        self.purchase_intention[brand_id] = 0.0
        self._realize_initial_state(brand_id, self._default_initial_cbbe(), stance=self.current_stance_global)

    def _default_initial_cbbe(self) -> Tuple[float, float, float]:
        """Driver-specific default perception, signed by the persona's initial stance."""
        defaults = {
            "performance_oriented": (0.3, 0.0, 0.0),
            "value_oriented": (0.2, 0.2, 0.0),
            "experience_oriented": (0.0, 0.1, 0.3),
            "social_identity_oriented": (0.0, 0.2, 0.2),
            "practical_safety_oriented": (0.3, 0.1, 0.0),
        }
        cap, val, emo = defaults.get(self.core_driver, (0.0, 0.0, 0.0))
        if self.current_stance_global == "opponent":
            return (-abs(cap) - 0.4, -abs(val) - 0.4, -abs(emo) - 0.4)
        if self.current_stance_global == "supporter":
            return (abs(cap) + 0.4, abs(val) + 0.4, abs(emo) + 0.4)
        return (cap * 0.1, val * 0.1, emo * 0.1)

    def _stance_thresholds(self) -> Tuple[float, float]:
        promoter_th, detractor_th = 0.4, -0.4
        thresholds = getattr(self.model, "stance_derivation_thresholds", None)
        if isinstance(thresholds, dict):
            try:
                promoter_th = float(thresholds.get("promoter_threshold", promoter_th))
                detractor_th = float(thresholds.get("detractor_threshold", detractor_th))
            except (TypeError, ValueError):
                pass
        return promoter_th, detractor_th

    def _realize_initial_state(self, brand_id: str, cbbe: Tuple[float, float, float], stance: Optional[str] = None) -> None:
        """Store the initial CBBE state and derive TPB intents consistent with the persona stance.

        The 5 x 3 persona design (Section 4.3) requires that supporters,
        neutrals, and opponents start in their stance band; an LLM or default
        perception rating is therefore signed and the advocacy intent is placed
        inside the band defined by the stance thresholds.
        """
        stance = stance if stance in STANCES else self.current_stance_global
        cap, val, emo = (max(-1.0, min(1.0, float(v))) for v in cbbe)
        if stance == "supporter":
            cap, val, emo = max(cap, 0.1), max(val, 0.1), max(emo, 0.1)
        elif stance == "opponent":
            cap, val, emo = min(cap, -0.1), min(val, -0.1), min(emo, -0.1)
        self.cbbe_state[brand_id] = {
            "capability_belief": cap,
            "value_resonance": val,
            "emotional_bond": emo,
        }
        vec = np.array([cap, val, emo], dtype=float)
        w_p = np.array(self._get_w_for_driver(self.core_driver, "purchase"), dtype=float)
        w_a = np.array(self._get_w_for_driver(self.core_driver, "advocacy"), dtype=float)
        purchase = float(_sigmoid(float(np.dot(w_p, vec))))
        advocacy = float(np.tanh(float(np.dot(w_a, vec))))
        promoter_th, detractor_th = self._stance_thresholds()
        draw = self.random.random() if self.random is not None else 0.5
        if stance == "supporter":
            advocacy = max(advocacy, min(1.0, promoter_th + 0.05 + 0.25 * draw))
        elif stance == "opponent":
            advocacy = min(advocacy, max(-1.0, detractor_th - 0.05 - 0.25 * draw))
        else:
            advocacy = max(detractor_th + 0.05, min(promoter_th - 0.05, advocacy))
        advocacy = max(-1.0, min(1.0, advocacy))
        self.intent_state[brand_id] = {"purchase_intent": purchase, "advocacy_intent": advocacy}
        self.stance_by_brand[brand_id] = self._derive_stance(advocacy)
        self.purchase_intention[brand_id] = purchase
        self.brand_sentiment[brand_id] = emo

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
                return {}
            user_input = (
                f"{prompt}\n\n"
                f"请严格只输出一个JSON对象，并满足如下模式要求（不要输出其他文字）：\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
            try:
                history, text = gemini_run(user_input, [])
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
                history, text = gemini_run(prompt, [])
                return str(text)
            except Exception:
                return ""

    def use_gemini_llm(self, model_name: str = "gemini-1.5-pro-latest") -> "Consumer":
        """为消费者Agent接入Gemini占位LLM（与其他Agent可独立配置）。"""
        self.llm = self.GeminiLLM(model_name=model_name)
        return self
    
    def step(self):
        """执行消费者的一个步骤
        
        消费者在每一步中会：
        1. 处理接收到的信息
        2. 更新对品牌的认知和态度
        3. 可能做出购买决策
        """
        # 处理消息队列中的消息
        self._process_messages()
        
        # 处理记忆中的信息
        self._process_memory()
        
        # 在状态更新后，根据宣传意图可能生成UGC
        try:
            # Check for model-level flag first (more robust), then instance-level flag
            manual_exec = getattr(self.model, "manual_consumer_execution", False)
            
            # Check if we are in a managed execution context
            # If strict_one_per_step is True, we should assume the external orchestrator calls _generate_ugc_for_brand explicitly
            # and we should NOT call _maybe_generate_ugc_all_brands here.
            strict_one = getattr(self, "strict_one_per_step", False)
            
            if not manual_exec and not strict_one:
                self._maybe_generate_ugc_all_brands()
        except Exception as e:
            print(f"[Consumer] Step UGC generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 可能的购买决策
        self._make_purchase_decision()
    
    def _process_messages(self):
        """处理消息队列中的消息"""
        for message in self.message_queue:
            message_type = message.get("message_type") or message.get("type")
            content = message.get("content")
            sender_role = message.get("sender_role")
            
            if message_type == "marketing":
                # 处理品牌营销信息
                brand_id = content.get("brand_id")
                sentiment_impact = content.get("sentiment_impact", 0)
                awareness_impact = content.get("awareness_impact", 0)
                info_vec = content.get("info_impact")  # 期望: {capability, value, emotion}
                
                # 根据发送者角色调整影响力
                current_s = self.stance_by_brand.get(brand_id, "neutral")
                if sender_role == "brand":
                    if current_s == "opponent":
                        impact_multiplier = 0.1  # Selective Exposure: Opponents ignore brand messages
                    else:
                        impact_multiplier = 1.0
                else:
                    impact_multiplier = 0.7
                
                # 应用影响力，考虑消费者的易受影响程度
                self._update_brand_perception(
                    brand_id,
                    sentiment_impact * impact_multiplier * self.influence_susceptibility,
                    awareness_impact * impact_multiplier * self.influence_susceptibility
                )

                # 新增：基于ELM信息影响向量的级联更新
                if brand_id and isinstance(info_vec, dict):
                    cap = float(info_vec.get("capability", 0.0))
                    val = float(info_vec.get("value", 0.0))
                    emo = float(info_vec.get("emotion", sentiment_impact))
                    verification_factor = float(content.get("verification_factor", 1.0)) if isinstance(content, dict) else 1.0
                    
                    # Exposure effects belong to Eq. (10); the belief update itself is Eq. (1).
                    self._cascade_update(brand_id, (cap, val, emo), scale=1.0,
                                         verification_factor=verification_factor)
            
            elif message_type == "official_post":
                 # Handle direct official post (similar to marketing but with different role check)
                 content_dict = (
                     content.get("content", {})
                     if isinstance(content, dict) and isinstance(content.get("content"), dict)
                     else (content if isinstance(content, dict) else {})
                 )
                 brand_id = content_dict.get("brand_id")
                 if brand_id:
                     info_vec = content_dict.get("info_impact", {})
                     verification_factor = float(content_dict.get("verification_factor", 1.0))
                     
                     self._cascade_update(brand_id,
                        (float(info_vec.get("capability", 0.0)), float(info_vec.get("value", 0.0)), float(info_vec.get("emotion", 0.0))),
                        scale=1.0,
                        verification_factor=verification_factor
                     )
            
            

            elif message_type == "benchmark":
                # t=0：研究者注入的考题文本，仅消费者接收
                exam_text = content.get("exam_text", "") if isinstance(content, dict) else str(content)
                target_brand_id = content.get("brand_id") if isinstance(content, dict) else None
                # Keep the focal brand stable across every consumer and run.
                # Only fall back to the first registered brand if an older caller
                # did not initialize ``model.focal_brand_id``.
                if target_brand_id is None:
                    target_brand_id = getattr(self.model, "focal_brand_id", None)
                if target_brand_id is None:
                    bids = [b.unique_id for b in self.model.agents_by_type.get("brand", [])]
                    if bids:
                        target_brand_id = sorted(str(bid) for bid in bids)[0]
                if target_brand_id is not None:
                    # 初始状态标定 + 初始UGC
                    try:
                        self.calibrate_initial_state(str(target_brand_id), str(exam_text))
                    except Exception:
                        pass
                    try:
                        self._generate_ugc_for_brand(str(target_brand_id), seed_text=str(exam_text))
                        self._skip_next_all_brands_generation = True
                    except Exception:
                        pass

            elif message_type in {"official_post", "ugc_post"}:
                # 对官方/UGC帖子进行级联影响更新（若含信息向量）
                info_vec = None
                brand_id = None
                if isinstance(content, dict):
                    info_vec = content.get("info_impact")
                    brand_id = content.get("brand_id")
                if brand_id and isinstance(info_vec, dict):
                    cap = float(info_vec.get("capability", 0.0))
                    val = float(info_vec.get("value", 0.0))
                    emo = float(info_vec.get("emotion", 0.0))
                    try:
                        v_factor = float(content.get("verification_factor", 1.0))
                        self._cascade_update(brand_id, (cap, val, emo), scale=1.0, verification_factor=v_factor)
                    except Exception:
                        pass
        
        # 清空消息队列
        self.message_queue = []
    
    def receive_message(self, message):
        """接收消息
        
        消费者会根据消息发送者的角色和影响力，以不同的方式处理消息。
        
        Args:
            message: 消息内容
        """
        super().receive_message(message)
    
    def _process_memory(self):
        """Awareness bookkeeping.

        CBBE/TPB states change only through Eq. (1); stances are re-derived
        from the current advocacy intent.
        """
        for brand_id in list(self.brand_awareness):
            self.brand_awareness[brand_id] *= 0.99
            intent = self.intent_state.get(brand_id)
            if intent is not None:
                self.stance_by_brand[brand_id] = self._derive_stance(float(intent.get("advocacy_intent", 0.0)))

    def _update_brand_perception(self, brand_id, sentiment_change, awareness_change):
        """更新对品牌的认知
        
        Args:
            brand_id: 品牌ID
            sentiment_change: 情绪变化
            awareness_change: 知名度变化
        """
        if brand_id in self.brand_awareness:
            # 更新品牌知名度，确保在[0, 1]范围内
            self.brand_awareness[brand_id] = min(1.0, max(0.0, self.brand_awareness[brand_id] + awareness_change))
            
            # 更新对品牌的情绪，确保在[-1, 1]范围内
            self.brand_sentiment[brand_id] = min(1.0, max(-1.0, self.brand_sentiment[brand_id] + sentiment_change))
            
            # 更新购买意向，基于知名度和情绪
            # 购买意向受正面情绪和品牌忠诚度的影响
            sentiment_factor = max(0, self.brand_sentiment[brand_id])  # 只有正面情绪增加购买意向
            self.purchase_intention[brand_id] = min(
                1.0,
                self.brand_awareness[brand_id] * 0.3 +  # 知名度的影响
                sentiment_factor * 0.5 +  # 情绪的影响
                self.loyalty * 0.2  # 忠诚度的影响
            )

    
    def _make_purchase_decision(self):
        """做出购买决策
        
        基于购买意向，消费者可能会做出购买决策。
        """
        # 简化版的购买决策，实际应该更复杂
        for brand_id, intention in self.intent_state.items():
            pi = float(intention.get("purchase_intent", self.purchase_intention.get(brand_id, 0.0)))
            if self.random.random() < pi * 0.1:  # 购买概率与购买意向成正比
                # 执行购买
                self._purchase(brand_id)
    
    def _purchase(self, brand_id):
        """执行购买
        
        Args:
            brand_id: 购买的品牌ID
        """
        # 记录购买行为
        purchase_info = {
            "consumer_id": self.unique_id,
            "brand_id": brand_id,
            "step": self.model.steps
        }
        
        # 通知模型有购买发生
        if hasattr(self.model, "record_purchase"):
            self.model.record_purchase(purchase_info)
        
        # 购买后提高对该品牌的忠诚度
        self.loyalty = min(1.0, self.loyalty + 0.05)
        
        # 购买后可能向邻居分享体验
        self._share_purchase_experience(brand_id)
    
    def _share_purchase_experience(self, brand_id):
        """分享购买体验
        
        消费者可能会向社交网络中的邻居分享自己的购买体验。
        
        Args:
            brand_id: 购买的品牌ID
        """
        # 只有一定概率会分享
        if self.random.random() < 0.3:
            neighbors = self.get_neighbors()
            
            # 分享的情感强度取决于对品牌的情绪
            sentiment = self.brand_sentiment[brand_id]
            
            # 创建分享内容
            content = {
                "brand_id": brand_id,
                "sentiment_impact": sentiment * 0.2,  # 分享的影响力较小
                "awareness_impact": 0.1  # 提高一点知名度
            }
            
            # 发送消息给邻居
            self.send_message(neighbors, "marketing", content)

    def _generate_ugc_for_brand(self, brand_id: str, seed_text: Optional[str] = None):
        """根据当前意图/立场生成一条UGC帖子并提交到平台（不直接推送）。"""
        # 生成用于LLM的提示模板（显式可见），便于后续接LLM
        try:
            dominant_topic = None
            if hasattr(self.model, "topic_keywords") and self.model.topic_keywords:
                dominant_topic = str(self.model.topic_keywords[0])
            prompt = self.render_consumer_social_prompt(
                brand_id=brand_id,
                content_type="new_post",
                in_reply_to_post_summary=seed_text or "",
                dominant_topic=dominant_topic or ""
            )
        except Exception as e:
            print(f"[Consumer] render_consumer_social_prompt failed for uid={self.unique_id}: {e}")
            import traceback
            traceback.print_exc()
            prompt = None
        if prompt is None:
            raise RuntimeError("Failed to build consumer prompt")

        def _fallback_text():
            stance = str(self.stance_by_brand.get(brand_id, self.current_stance_global) or "neutral")
            driver = str(self.core_driver or "value_oriented")
            seed_snippet = (seed_text or "")[:40]
            mood = {
                "supporter": "支持但希望及时公布进展",
                "neutral": "观望并提出疑问",
                "opponent": "质疑安全与透明",
            }.get(stance, "关注进展")
            driver_focus = {
                "value_oriented": "性价比",
                "performance_oriented": "性能数据",
                "experience_oriented": "体验细节",
                "social_identity_oriented": "口碑影响",
                "practical_safety_oriented": "安全细节",
            }.get(driver, "产品信息")
            return f"{seed_snippet}…想了解更多{driver_focus}，{mood}。"

        # ==== Fallback: 保证在无LLM或调用失败时也能生成一条合理发言 ====
        model_name = str(getattr(self, "llm_model_name", DEFAULT_MODEL) or DEFAULT_MODEL)
        offline_requested = model_name.strip().lower() in {"offline", "heuristic", "none"}
        network_disabled = os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", "1") != "0"
        if gemini_run is None or offline_requested or network_disabled:
            response_text = _fallback_text()
        else:
            try:
                # Use empty history because the prompt contains all necessary context
                # including persona, state, and recent history.
                # The agent is stateless from the LLM's perspective.
                history, response_text = gemini_run(prompt, [], model_name=model_name)
            except Exception:
                response_text = _fallback_text()

        def _sanitize_text(s: str) -> str:
            try:
                import re
                s = re.sub(r"#\S+", "", s)
                s = re.sub(r"[\[\]{}()<>【】（）]", "", s)
                # s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff，。！？、；：\-\s]", "", s) # Too aggressive
                s = s.replace("XX", "").replace("XXX", "").replace("某个", "")
                s = re.sub(r"\s{2,}", " ", s).strip()
                if s and s[-1] not in "。?!！？":
                    s += "。"
                return s
            except Exception:
                return s

        text = _sanitize_text(str(response_text).strip())
        if not text:
            raise RuntimeError("Empty response from Gemini API for consumer post")
        tags = list(self.model.topic_keywords or [])
        # 准备 info_impact (映射 CBBE 状态到 ELM 维度)
        cbbe = self.cbbe_state.get(brand_id, {})
        info_impact = {
            "capability": cbbe.get("capability_belief", 0.0),
            "value": cbbe.get("value_resonance", 0.0),
            "emotion": cbbe.get("emotional_bond", 0.0),
        }

        # Record exactly once through the platform bus.  ``record_message``
        # creates the canonical JSONL post and retains all mechanism metadata.
        try:
            if hasattr(self.model, "record_message"):
                self.model.record_message({
                    "sender_id": self.unique_id,
                    "sender_role": "consumer",
                    "message_type": "ugc_post",
                    "content": {
                        "post_id": None,
                        "author_id": self.unique_id,
                        "authorPersonality": {"driver": self.core_driver, "initial_stance": self.current_stance_global},
                        "authorStance": self.stance_by_brand.get(brand_id, self.current_stance_global),
                        "contentType": "new_post",
                        "content": text,
                        "inReplyTo": None,
                        "topicTags": tags,
                        "brand_id": brand_id,
                        "info_impact": info_impact,
                        "verification_factor": 1.0,
                    },
                })
        except Exception:
            pass

    def render_consumer_social_prompt(
        self,
        brand_id: str,
        content_type: str = "new_post",
        in_reply_to_post_summary: str = "",
        dominant_topic: Optional[str] = None,
    ) -> str:
        """渲染用户Agent的社交媒体生成提示模板。

        Args:
            brand_id: 品牌ID，用于读取该品牌维度的状态向量
            content_type: 'new_post' 或 'comment'
            in_reply_to_post_summary: 若为评论场景，传入被回复帖子摘要
            dominant_topic: 当前热点话题（可选，默认取模型topic_keywords首项）

        Returns:
            完整的提示字符串，供LLM直接使用。
        """
        persona_example = PROTOTYPE_LABELS.get(
            (self.core_driver, self.current_stance_global), self.prototype_label
        )
        cs = self.cbbe_state.get(
            brand_id,
            {"capability_belief": 0.0, "value_resonance": 0.0, "emotional_bond": 0.0},
        )
        is_ = self.intent_state.get(brand_id, {"advocacy_intent": 0.0})
        stance = self.stance_by_brand.get(brand_id, self.current_stance_global)

        if dominant_topic is None:
            try:
                if hasattr(self.model, "topic_keywords") and self.model.topic_keywords:
                    dominant_topic = str(self.model.topic_keywords[0])
                else:
                    dominant_topic = ""
            except Exception:
                dominant_topic = ""

        # Determine Event Guidance based on topic/context
        event_guidance = ""
        dt_lower = dominant_topic.lower() if dominant_topic else ""
        
        # Event 1: Accident (Check keywords like "accident", "crash", "collision", "death", "SU7", "crash")
        # In Chinese: "事故", "撞", "起火", "死", "伤"
        if any(k in dt_lower for k in ["事故", "撞", "起火", "死", "伤", "accident", "crash"]):
             event_guidance = (
                 "Context Note: A serious car accident has occurred. "
                 "If you are 'Safety Oriented' or 'Practical', you are VERY concerned/scared. "
                 "If you are 'Performance Oriented', you might defend the car unless evidence is clear. "
                 "If you are 'Value Oriented', you worry about resale value. "
                 "Initial reaction is shock/fear. Over time, look for official explanation."
             )
        # Event 2: Recall (Check keywords like "recall", "defect", "OTA", "upgrade", "fix")
        # In Chinese: "召回", "升级", "备案", "隐患"
        elif any(k in dt_lower for k in ["召回", "备案", "隐患", "recall"]):
             current_step = 0
             if hasattr(self.model, "steps"):
                 current_step = self.model.steps
             elif hasattr(self.model, "schedule") and hasattr(self.model.schedule, "steps"):
                 current_step = self.model.schedule.steps
             
             # Dynamic guidance based on TRUST (Value Congruence)
             # If trust is low (e.g. Specious strategy backfiring), explicitly prompt for skepticism.
             trust_level = float(cs.get("value_resonance", 0.0))
             
             # Check for recent low verification factor
             last_v = 1.0
             if hasattr(self, "last_verification_status"):
                 last_v = self.last_verification_status.get(brand_id, 1.0)

             if trust_level < -0.1 or last_v < 0.8:
                 event_guidance = (
                     "Context Note: You DO NOT TRUST the brand's latest statement. "
                     "You suspect they are lying, covering up, or using fake evidence. "
                     "Use words like 'Fake', 'Liar', 'Cover-up', '忽悠', '智商税', '骗子', '不信'. "
                     "You feel angry and manipulated."
                 )
             elif current_step < 2:
                 event_guidance = (
                     "Context Note: The brand announced a RECALL. "
                     "Supporters: Praise the responsibility, but admit it's annoying. "
                     "Others: 'Recall = Product Defect'. Be suspicious. Why now? Is it safe to drive? "
                     "Sentiment: Mixed/Cautious."
                 )
             else:
                 event_guidance = (
                     "Context Note: The recall discussion is heating up. "
                     "Competitors are highlighting the danger. "
                     "The defect involves SUDDEN FAILURE. It is NOT just a software bug. "
                     "You feel BETRAYED and SCARED. "
                     "Even if you liked the brand, now you worry about resale value and safety. "
                     "CRITICIZE the quality control. 'How did this pass testing?'"
                 )

        prompt = CONSUMER_SOCIAL_PROMPT_TEMPLATE.format(
            core_driver=self.core_driver,
            initial_stance=self.current_stance_global,
            persona_example=persona_example,
            competenceBelief_score=round(float(cs.get("capability_belief", 0.0)), 3),
            valueCongruence_score=round(float(cs.get("value_resonance", 0.0)), 3),
            emotionalConnection_score=round(float(cs.get("emotional_bond", 0.0)), 3),
            advocacyIntent_score=round(float(is_.get("advocacy_intent", 0.0)), 3),
            current_stance=stance,
            contentType=content_type,
            inReplyTo_post_summary=in_reply_to_post_summary or "",
            dominant_topic=dominant_topic,
            event_guidance=event_guidance
        )
        return prompt

    def _maybe_generate_ugc_all_brands(self):
        """在t.2阶段：根据最新宣传意图决定是否生成UGC。"""
        if getattr(self, "_skip_next_all_brands_generation", False):
            self._skip_next_all_brands_generation = False
            # start_benchmark invokes this hook at model step 0; at the first
            # real timestep the flag is cleared and generation proceeds.
            if int(getattr(self.model, "steps", 0) or 0) == 0:
                return
        # 强制每位消费者每轮至少对所有关注品牌发一条内容，满足“每个Agent每一轮都要发言”需求
        for bid, _ in self.intent_state.items():
            try:
                self._generate_ugc_for_brand(bid)
            except Exception as e:
                print(f"[Consumer] _generate_ugc_for_brand failed for {bid}: {e}")
                import traceback
                traceback.print_exc()

    def force_generate_ugc_for_brand(self, brand_id: str):
        try:
            self._generate_ugc_for_brand(brand_id)
        except Exception:
            pass

    def force_generate_ugc_one(self, preferred_brand_id: str | None = None):
        try:
            bid = preferred_brand_id
            if not bid:
                # 选择任一已存在的品牌意图键
                keys = list(self.intent_state.keys())
                bid = keys[0] if keys else None
            if bid:
                self._generate_ugc_for_brand(bid)
        except Exception:
            pass

    # ======= Eq. (1)/(11): CBBE update -> TPB intents -> stance =======
    def _update_rate(self) -> float:
        """λ_u in (0, 1): base rate x global decay factor x driver group factor."""
        hp = getattr(self.model, "hyperparams", {}) or {}
        decay = float(hp.get("memory_decay_global_lambda", hp.get("memory_decay_lambda", 1.0)))
        gamma_map = getattr(self.model, "group_influence_gamma", {}) or {}
        group = 1.0
        if isinstance(gamma_map, dict):
            try:
                group = float(gamma_map.get(self.core_driver, 1.0))
            except (TypeError, ValueError):
                group = 1.0
        return float(min(max(self.learning_rate_lambda * decay * group, 1e-6), 1.0))

    def _cascade_update(self, brand_id: str, info_vector: Tuple[float, float, float], scale: float = 1.0, verification_factor: float = 1.0):
        """Consumer state update, Eq. (1)/(11) of the paper:

            V_t = (1 - λ_u) V_{t-1} + λ_u · γ_t · tanh(X_t),   X_t = β_driver · x_t

        where ``x_t`` is the message-derived impact vector (capability, value,
        emotion), ``β_driver`` the consumer's driver-specific sensitivity matrix
        and ``γ_t ∈ [0, 1]`` the evidence-consistency gate, applied as a
        multiplicative attenuation.  Gaussian noise is added only when
        ``memory_volatility`` > 0 is configured.
        """
        try:
            gamma = float(verification_factor)
        except (TypeError, ValueError):
            gamma = 1.0
        if not np.isfinite(gamma):
            gamma = 1.0
        gamma = max(0.0, min(1.0, gamma))
        if not hasattr(self, "last_verification_status"):
            self.last_verification_status = {}
        self.last_verification_status[brand_id] = gamma

        if brand_id not in self.cbbe_state:
            self.register_brand(brand_id)

        beta = np.array(self._get_beta_for_driver(self.core_driver), dtype=float)
        x = np.nan_to_num(np.array(info_vector, dtype=float) * float(scale), nan=0.0, posinf=1.0, neginf=-1.0)
        x = np.clip(x, -1.0, 1.0)
        impact = beta @ x

        lam = self._update_rate()
        prev = np.array([
            self.cbbe_state[brand_id]["capability_belief"],
            self.cbbe_state[brand_id]["value_resonance"],
            self.cbbe_state[brand_id]["emotional_bond"],
        ], dtype=float)
        updated = (1.0 - lam) * prev + lam * gamma * np.tanh(impact)
        if self.memory_volatility > 0 and self.random is not None:
            updated = updated + np.array([self.random.gauss(0.0, self.memory_volatility) for _ in range(3)])
        updated = np.clip(updated, -1.0, 1.0)

        self.cbbe_state[brand_id]["capability_belief"] = float(updated[0])
        self.cbbe_state[brand_id]["value_resonance"] = float(updated[1])
        self.cbbe_state[brand_id]["emotional_bond"] = float(updated[2])

        # TPB intents (release mapping): purchase = σ(w_p·V), advocacy = tanh(w_a·V)
        w_p = np.array(self._get_w_for_driver(self.core_driver, "purchase"), dtype=float)
        w_a = np.array(self._get_w_for_driver(self.core_driver, "advocacy"), dtype=float)
        purchase_intent = float(_sigmoid(float(np.dot(w_p, updated))))
        advocacy_intent = float(np.tanh(float(np.dot(w_a, updated))))
        self.intent_state[brand_id]["purchase_intent"] = purchase_intent
        self.intent_state[brand_id]["advocacy_intent"] = advocacy_intent
        self.purchase_intention[brand_id] = purchase_intent
        self.brand_sentiment[brand_id] = float(updated[2])
        self.stance_by_brand[brand_id] = self._derive_stance(advocacy_intent)

    def _derive_stance(self, advocacy_intent: float) -> str:
        promoter_th, detractor_th = self._stance_thresholds()
        if advocacy_intent > promoter_th:
            return "supporter"
        if advocacy_intent < detractor_th:
            return "opponent"
        return "neutral"

    def _get_beta_for_driver(self, driver: str) -> List[List[float]]:
        """ELM β for the driver: platform calibration if supplied, else the release defaults."""
        if hasattr(self.model, "get_calibration_beta"):
            try:
                beta = self.model.get_calibration_beta(driver)
            except Exception:
                beta = None
            if beta is not None:
                return beta
        return [list(row) for row in DEFAULT_BETA.get(driver, DEFAULT_BETA["value_oriented"])]

    def _get_w_for_driver(self, driver: str, kind: str) -> List[float]:
        """TPB weights for the driver; kind in {"purchase", "advocacy"}."""
        if hasattr(self.model, "get_calibration_w"):
            try:
                w = self.model.get_calibration_w(driver, kind)
            except Exception:
                w = None
            if w is not None:
                return w
        return list(DEFAULT_W.get(driver, DEFAULT_W["value_oriented"])[kind if kind == "purchase" else "advocacy"])

    # ======= 初始状态标定（t=0） =======
    def calibrate_initial_state(self, brand_id: str, exam_text: str, llm: Optional[Any] = None):
        """使用LLM对“事件前”文本进行打分，初始化CBBE三维感知状态。"""
        # Try to use the standard gemini_run if available, as fallback to object method
        cap, val, emo = 0.0, 0.0, 0.0
        success = False

        prompt_text = (
                f"你是一个[{self.prototype_label}]。请阅读以下关于品牌的背景事件文本，并根据你的角色设定，评估你对该品牌的初始感知状态。\n"
                f"背景文本：{exam_text}\n\n"
                "请从以下三个维度进行打分（范围 -1.0 到 1.0，其中 -1.0 为极度负面/不信任，0.0 为中立，1.0 为极度正面/信任）：\n"
                "1. capability_belief (能力信念): 品牌的技术实力、产品质量、安全性。\n"
                "2. value_resonance (价值共鸣): 品牌的价值观、责任感、是否符合你的期望。\n"
                "3. emotional_bond (情感联结): 你对品牌的喜爱程度、亲近感。\n\n"
                "注意：\n"
                "- 如果是由于不可抗力或偶然因素导致的事故，但品牌态度诚恳，可能不需要打极低分。\n"
                "- 如果是品牌主动召回（Recall/升级），请注意：\n"
                "  * 虽然体现了责任感（Value Resonance可能>0），但这也直接证实了产品存在【缺陷】（Capability Belief 必须大幅下降，<0）。\n"
                "  * 对于看重安全（Safety-Oriented）的用户，这甚至是背叛，情感联结（Emotional Bond）也会受损。\n"
                "  * 不要盲目给高分，这是产品质量事故！\n"
                "- 请务必模仿真实消费者的心理活动，不要只给出极端值。\n\n"
                "请输出标准JSON格式: {\"capability_belief\": float, \"value_resonance\": float, \"emotional_bond\": float}"
            )

        try:
            # Method A: Try using gemini_run directly (most reliable if imported)
            model_name = getattr(self, "llm_model_name", DEFAULT_MODEL)
            offline = (
                os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", "1") != "0"
                or str(model_name).strip().lower() in {"offline", "heuristic", "none"}
            )
            if gemini_run is not None and not offline:
                _, response_text = gemini_run(prompt_text, [], model_name=model_name, response_format="json")
                
                # Parse JSON from text
                import json
                import re
                
                # Try to find JSON block
                json_match = re.search(r'\{.*\}', str(response_text), re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    cap = float(np.clip(data.get("capability_belief", 0.0), -1.0, 1.0))
                    val = float(np.clip(data.get("value_resonance", 0.0), -1.0, 1.0))
                    emo = float(np.clip(data.get("emotional_bond", 0.0), -1.0, 1.0))
                    success = True
            
            # Method B: Try existing llm object if Method A didn't run or failed
            if not success:
                if llm is None:
                    if getattr(self, "llm", None):
                        llm = self.llm
                    elif hasattr(self.model, "llm"):
                        llm = self.model.llm
                
                if llm is not None and hasattr(llm, "generate_with_json"):
                    schema = {
                        "type": "object",
                        "properties": {
                            "capability_belief": {"type": "number"},
                            "value_resonance": {"type": "number"},
                            "emotional_bond": {"type": "number"},
                        },
                        "required": ["capability_belief", "value_resonance", "emotional_bond"],
                        "additionalProperties": False,
                    }
                    result = llm.generate_with_json(prompt_text, json_schema=schema, temperature=0.2, max_tokens=400)
                    cap = float(np.clip(result.get("capability_belief", 0.0), -1.0, 1.0))
                    val = float(np.clip(result.get("value_resonance", 0.0), -1.0, 1.0))
                    emo = float(np.clip(result.get("emotional_bond", 0.0), -1.0, 1.0))
                    success = True
                    
        except Exception as e:
            # Fallback will handle it
            pass

        if not success:
            # Offline: driver-specific defaults signed by the persona stance.
            cap, val, emo = self._default_initial_cbbe()

        # Realize the persona's initial stance (supporter / neutral / opponent)
        # whether the perception came from the LLM or from the defaults.
        self._realize_initial_state(brand_id, (cap, val, emo), stance=self.current_stance_global)
