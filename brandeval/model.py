"""品牌营销多智能体仿真模型"""

from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import numpy as np
import threading

from mesa import Model
# Optional: DataCollector depends on pandas; wrap import to avoid hard failure
try:
    from mesa.datacollection import DataCollector
except Exception:
    DataCollector = None
from mesa.discrete_space import Network

# 兼容：如果缺少 Mesa 的 RandomActivation，则使用本地简易调度器
try:
    from mesa.time import RandomActivation  # type: ignore
except Exception:
    class RandomActivation:
        """本地简易随机调度器，用于兼容缺失的 mesa.time.RandomActivation。

        提供 add(agent) 与 step()，在每步中随机打乱顺序并调用 agent.step()。
        """

        def __init__(self, model):
            self.model = model
            self._agents = []

        def add(self, agent):
            self._agents.append(agent)

        def step(self):
            try:
                rnd = getattr(self.model, "random", None)
                if rnd:
                    rnd.shuffle(self._agents)
                else:
                    import random as _r
                    _r.shuffle(self._agents)
            except Exception:
                pass
            for a in list(self._agents):
                try:
                    a.step()
                except Exception:
                    pass

from brandeval.agents.consumer import Consumer
from brandeval.agents.brand import Brand
# KOL模块已移除，模型聚焦品牌、消费者、竞争对手与平台
# 说明：网络工具模块已清理；模型内部已提供 _create_social_network 实现。


def public_text(message: Any) -> str:
    """The platform-visible text of a message: never impact vectors, prompts, or latent states."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, dict):
        content = content.get("content", content.get("text", ""))
    return str(content or "")


class BrandEvalModel(Model):
    """品牌营销多智能体仿真模型
    
    这个模型模拟了品牌方、消费者与竞争对手在营销场景中的交互。
    模型包含一个社交网络，用于模拟信息传播，以及各种指标用于评估营销策略的效果。
    """
    
    def __init__(
        self,
        num_consumers: int = 100,
        num_brands: int = 3,
        network_type: str = "small_world",
        network_params: Dict = {"k": 4, "p": 0.1},
        seed: Optional[int] = None,
        population_spec: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        """初始化模型
        
        Args:
            num_consumers: 消费者数量
            num_brands: 品牌数量
            network_type: 社交网络类型 ("small_world", "scale_free", "random")
            network_params: 社交网络参数
            seed: 随机种子
        """
        super().__init__(seed=seed)
        # 调度器
        self.schedule = RandomActivation(self)
        
        # 模型参数
        self.num_consumers = num_consumers
        self.num_brands = num_brands
        self.network_type = network_type
        self.network_params = network_params
        # Mesa 3 advances ``steps`` in its wrapped scheduler before calling the user step.
        # Keep a focal brand id separate from the scheduler counter.
        self.focal_brand_id: Optional[str] = None
        self._message_lock = threading.RLock()
        self._message_fingerprints_current: set[tuple] = set()

        # 平台环境：中心历史与消息总线
        self.history: List[Dict[str, Any]] = []
        self._message_buffer_current: List[Dict[str, Any]] = []
        self._message_buffer_step: List[Dict[str, Any]] = []  # Buffer for metrics in the current step
        self._message_buffer_prev: List[Dict[str, Any]] = []
        self._posts_current: List[Dict[str, Any]] = []
        self._last_step_messages: List[Dict[str, Any]] = []
        self._last_step_posts: List[Dict[str, Any]] = []
        self._next_post_id: int = 1
        self.memory_base: List[Dict[str, Any]] = []
        self.newly_archived_posts: List[str] = []
        self._top_posts_by_weight: List[Dict[str, Any]] = []
        self._strategic_actions_current: List[Dict[str, Any]] = []
        self.jsonl_records: List[Dict[str, Any]] = []
        self.bri_weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        # 信息分发参数（注意力稀缺与饱和效应）
        self.base_push_p: float = 0.1
        self.push_c1: float = 0.1
        self.push_c2: float = 0.1

        
        self.topic_keywords: List[str] = []
        self.brand_keywords: List[str] = []
        self.competitor_keywords: List[str] = []
        self._last_dominant_owner: Optional[str] = None
        
        # 按类型存储智能体（由 Mesa Model 管理），无需重新赋值
        # 使用 self.agents_by_type 读写其内部字典即可
        
        # 外部配置存储
        self._population_spec = population_spec or {}
        self._calibration_beta = (calibration or {}).get("beta", {})
        self._calibration_w = (calibration or {}).get("w", {})

        # BrandEval Eq. (12) fixes the three CBBE weights to 1/3 each.
        # Regression coefficients remain available for state transitions, but
        # must not silently redefine the reported BRI metric.
        self.bri_weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        # 组影响与立场阈值
        self.hyperparams = hyperparams or {}
        hp = self.hyperparams
        self.group_influence_gamma = dict(hp.get("group_influence_gamma", {}))
        # Stance groups coincide with NPS promoter/detractor groups (paper
        # scale > 0.7 / < 0.3  <=>  internal tanh scale > 0.4 / < -0.4).
        self.stance_derivation_thresholds = dict(hp.get("stance_derivation_thresholds", {
            "promoter_threshold": 0.4,
            "detractor_threshold": -0.4,
        }))
        # 平台扩散与热度衰减
        try:
            # P1: Platform Exposure Base Rate
            self.base_push_p = float(hp.get("platform_base_p", hp.get("base_push_p", 0.1)))
            
            # P2: Influence Weight
            self.push_c1 = float(hp.get("platform_virality_c1", hp.get("push_c1", 0.1)))
            
            # P3: Popularity Weight
            self.push_c2 = float(hp.get("platform_topic_c2", hp.get("push_c2", 0.1)))
            
            # Topic-heat decay per round (release default 0.9; not stated in the paper)
            self.heat_decay_gamma = float(hp.get("platform_heat_decay_gamma", hp.get("platform_heat_decay_factor", 0.9)))
        except Exception:
            pass
        self.backfire_rounds: int = 0
        self.metric_rounds: int = 0
        
        self.global_heat = 0.0
        # Topic-specific heat is tracked separately from the aggregate heat so
        # exposure can use the topic attached to each message when available.
        self.topic_heat: Dict[str, float] = {}

        
        # 购买记录
        self.purchase_history = []
        
        # 创建社交网络
        self.network = self._create_social_network()
        
        # 创建网络空间
        self.space = Network(self.network, capacity=None, random=self.random)
        
        # 创建数据收集器（在无 pandas 环境下优雅回退）
        if DataCollector is not None:
            self.datacollector = DataCollector(
                model_reporters={
                    "Brand_Awareness": self._calculate_brand_awareness,
                    "Consumer_Sentiment": self._calculate_consumer_sentiment,
                    "Purchase_Intention": self._calculate_purchase_intention,
                    "Network_Density": lambda m: nx.density(self.network),
                    "Sales_Volume": self._calculate_sales_volume,
                    "Brand_Reputation": self._calculate_brand_reputation,
                    # 已移除KOL影响力指标
                },
                agent_reporters={
                    "Role": "role",
                    "Sentiment": "sentiment",
                    "Purchase_Intention": "purchase_intention",
                }
            )
        else:
            self.datacollector = None
        
        # 创建智能体
        self._create_agents()
        
        # 收集初始数据（若启用数据收集器）
        if self.datacollector is not None:
            self.datacollector.collect(self)

    # ================= 平台环境接口 =================
    def inject_messages(self, messages: List[Dict[str, Any]]):
        """外部注入消息（如来自上一轮的品牌消息）。"""
        for msg in messages:
            self.record_message(msg)
        # 立即分发
        self.distribute_messages()

    def distribute_messages(self):
        """将当前缓冲区的消息分发给所有消费者。"""
        # 简单广播模型：所有消费者接收所有公共消息
        # 实际逻辑可根据关注关系优化
        consumers = self.agents_by_type.get("consumer", [])
        for msg in self._message_buffer_current:
            for c in consumers:
                if hasattr(c, "message_queue"):
                    c.message_queue.append(msg)
        
        # 清空当前缓冲 (移入历史)
        self.history.extend(self._message_buffer_current)
        self._message_buffer_current = []
        self._enforce_consumer_feed_capacity()

    def _enforce_consumer_feed_capacity(self) -> None:
        """Keep at most C public messages per consumer, prioritizing broadcasts."""
        capacity = max(0, int(self.hyperparams.get("feed_capacity", 20)))
        for consumer in self.agents_by_type.get("consumer", []):
            queue = list(getattr(consumer, "message_queue", []) or [])
            if len(queue) <= capacity:
                continue
            ranked = sorted(
                enumerate(queue),
                key=lambda item: (
                    0 if str(item[1].get("sender_role", item[1].get("role", ""))).lower() == "brand" else 1,
                    -item[0],
                ),
            )
            consumer.message_queue = [message for _, message in ranked[:capacity]]

    def _record_message_unlocked(self, message: Dict[str, Any]):
        """Record one message while preserving the structured post payload.

        Consumers need ``info_impact`` and ``verification_factor`` for the ELM
        and evidence-gating updates.  Earlier code flattened nested post
        payloads to plain text, silently discarding those fields.  The payload
        remains a dict with ``content`` holding the display text; text-only
        callers are normalized into the same shape.
        """
        if not isinstance(message, dict):
            return
        msg = dict(message)
        mtype = str(msg.get("message_type", "")).lower()
        if mtype in {"ugc_post", "official_post", "new_post"}:
            original = msg.get("content")
            payload = dict(original) if isinstance(original, dict) else {}
            text = str(payload.get("content") or payload.get("text") or (original if original is not None else ""))
            content_type = str(payload.get("contentType") or "new_post")
            in_reply_to = payload.get("inReplyTo")
            brand_id = payload.get("brand_id", msg.get("brand_id"))
            author_personality = payload.get("authorPersonality")
            author_stance = payload.get("authorStance")
            topic_tags = payload.get("topicTags")
            # Preserve all mechanism metadata rather than dropping unknown keys.
            payload["content"] = text
            payload.setdefault("contentType", content_type)
            if brand_id is not None:
                payload["brand_id"] = brand_id
                msg["brand_id"] = brand_id
            if not topic_tags and self.topic_keywords:
                import re
                topic_tags = [k for k in self.topic_keywords
                              if re.search(re.escape(k), text, flags=re.IGNORECASE)]
            payload["topicTags"] = list(topic_tags or [])
            msg["content"] = payload

            # Reuse a supplied post id when present; otherwise generate one.
            post_id = msg.get("post_id") or payload.get("post_id")
            if not post_id:
                post_id = f"post_{self._next_post_id}"
                self._next_post_id += 1
            msg["post_id"] = post_id
            # Do not re-ingest the same post when a caller hands the previous
            # buffer back to ``inject_messages``.
            existing_ids = {m.get("post_id") for m in (self._message_buffer_current + self._message_buffer_prev)}
            if post_id in existing_ids:
                return
            fingerprint = (msg.get("sender_id"), msg.get("sender_role"), mtype,
                           str(brand_id or ""), text)
            if fingerprint in self._message_fingerprints_current:
                return
            self._message_fingerprints_current.add(fingerprint)
            self._posts_current.append({
                "postId": post_id,
                "authorId": msg.get("sender_id"),
                "authorRole": msg.get("sender_role"),
                "authorPersonality": author_personality,
                "authorStance": author_stance,
                "contentType": content_type,
                "content": text,
                "inReplyTo": in_reply_to,
                "topicTags": list(topic_tags or []),
                "brand_id": brand_id,
                "info_impact": payload.get("info_impact", {}),
                "verification_factor": payload.get("verification_factor", 1.0),
            })
        self._message_buffer_current.append(msg)
        self._message_buffer_step.append(msg)

    def record_message(self, message: Dict[str, Any]):
        """Record a message atomically when consumer workers run in parallel."""
        with self._message_lock:
            self._record_message_unlocked(message)

    def register_topic_keywords(self, keywords: List[str]):
        self.topic_keywords = list(keywords or [])

    def _message_topics(self, message: Dict[str, Any]) -> List[str]:
        """Return topic labels attached to a message, with a keyword fallback."""
        content = message.get("content") if isinstance(message, dict) else None
        tags = message.get("topicTags") or message.get("topic_tags")
        if not tags and isinstance(content, dict):
            tags = content.get("topicTags") or content.get("topic_tags")
        if isinstance(tags, str):
            tags = [tags]
        topics = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        if topics:
            return topics
        text = content.get("content", "") if isinstance(content, dict) else content
        text = str(text or "")
        if not text or not self.topic_keywords:
            return []
        import re
        return [k for k in self.topic_keywords if re.search(re.escape(k), text, flags=re.IGNORECASE)]

    def register_brand_competitor_keywords(self, brand_keywords: List[str], competitor_keywords: List[str]):
        self.brand_keywords = list(brand_keywords or [])
        self.competitor_keywords = list(competitor_keywords or [])

    def _compute_effective_push_p(self, influence: float, heat: float) -> float:
        """Eq. (9): P_push = clip[0,1](P_base [1 + c1 ln(I_s + 1) + c2 ln(H_k + 1)])."""
        import math
        p = self.base_push_p
        return max(0.0, min(1.0, p * (1.0 + self.push_c1 * math.log(max(0.0, influence) + 1.0) + self.push_c2 * math.log(max(0.0, heat) + 1.0))))

    @staticmethod
    def _retention_susceptibility(consumer: Any) -> float:
        """s_u in Eq. (10): the consumer's retention susceptibility, drawn U(0,1) at creation."""
        try:
            value = float(getattr(consumer, "influence_susceptibility", 1.0))
        except (TypeError, ValueError):
            value = 1.0
        return max(0.0, min(1.0, value))

    def _message_valence(self, message: Dict[str, Any]) -> int:
        """+1 when a message favours the focal brand, -1 when it opposes it, 0 otherwise."""
        role = str(message.get("sender_role", "")).lower()
        if role == "brand":
            return 1
        if role == "competitor":
            return -1
        content = message.get("content")
        payload = content if isinstance(content, dict) else {}
        stance = str(payload.get("authorStance") or "").lower()
        target = payload.get("brand_id", message.get("brand_id"))
        sign = {"supporter": 1, "opponent": -1}.get(stance, 0)
        if sign == 0:
            return 0
        about_focal = self.focal_brand_id is None or str(target) == str(self.focal_brand_id)
        return sign if about_focal else -sign

    def _stance_alignment(self, message: Dict[str, Any], consumer: Any) -> float:
        """a_{m,u} in Eq. (10): 1 for aligned, ``stance_alignment_neutral`` when either side is neutral,
        ``stance_alignment_opposed`` for conflicting stances (release operationalization)."""
        valence = self._message_valence(message)
        stance_map = getattr(consumer, "stance_by_brand", {}) or {}
        stance = str(stance_map.get(self.focal_brand_id, getattr(consumer, "current_stance_global", "neutral"))).lower()
        sign = {"supporter": 1, "opponent": -1}.get(stance, 0)
        if valence == 0 or sign == 0:
            return float(self.hyperparams.get("stance_alignment_neutral", 0.75))
        if valence == sign:
            return 1.0
        return float(self.hyperparams.get("stance_alignment_opposed", 0.5))

    def _get_agent_by_id(self, uid: Any):
        # 遍历所有类型以查找代理
        for lst in self.agents_by_type.values():
            for a in lst:
                if getattr(a, "unique_id", None) == uid:
                    return a
        return None

    

    def _compute_sov_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates Share of Voice (SoV) as 'Share of Discussion'.
        
        Definition:
            SoV = (Brand Official Posts + Brand-Related UGC) / Total Messages
            
        Note on Numerical Scale:
            In this simulation, 'Total Messages' represents the volume of conversation 
            generated *within the specific topic context* of the event. It does not 
            include the vast 'background noise' of the entire internet. 
            Therefore, SoV values are expected to be high (e.g., 50%-80%) for the 
            focal brand during a relevant event, as most generated content is 
            reactionary to the scenario. This differs from a 'Global SoV' on a 
            real social platform where a single brand might have <1% share against 
            all unrelated traffic.
            
        Logic:
        1. Identifies messages from the Brand (Official).
        2. Identifies UGC that explicitly tags/mentions the Brand (via 'brand_id' metadata).
        3. Aggregates both as the Brand's total volume.
        """
        # 1. Identify Brands
        brands = self.agents_by_type.get("brand", [])
        if not brands:
            return {
                "Brand_Current_SoV": 0.0,
                "Competitor_Current_SoV": 0.0,
            }
            
        # Assume Brand 0 is Focal
        focal_brand = brands[0]
        focal_id = self.focal_brand_id or getattr(focal_brand, "unique_id", "brand_0")
        
        # Identify Competitor ID
        comp_id = "competitor_dummy"
        competitors = self.agents_by_type.get("competitor", [])
        if competitors:
             comp_id = getattr(competitors[0], "unique_id", "competitor_dummy")
        elif len(brands) > 1:
             comp_id = getattr(brands[1], "unique_id", "competitor_dummy")

        # 2. Initialize Counters
        focal_vol = 0
        comp_vol = 0
        
        total_vol = len(messages)
        
        # 3. Process Messages
        for m in messages:
            role = str(m.get("sender_role", "")).lower()
            sender_id = m.get("sender_id")
            content_brand_id = m.get("brand_id") # Injected by record_message
            
            # Case A: Brand's Own Voice
            if role == "brand":
                if sender_id == focal_id:
                    focal_vol += 1
                elif sender_id == comp_id:
                    comp_vol += 1
                continue
                
            # Case B: Competitor's Own Voice
            if role == "competitor":
                comp_vol += 1
                continue
                
            # Case C: Consumer Voice (or others)
            # Identify Target Brand via Metadata
            target_id = None
            if content_brand_id:
                target_id = content_brand_id
            
            # Logic: If target is identified, attribute volume
            if target_id == focal_id:
                focal_vol += 1
            elif target_id == comp_id or (comp_id != "competitor_dummy" and target_id == comp_id):
                comp_vol += 1

        # 4. Calculate Metrics
        # Denominator is Total Volume (All messages in step)
        brand_sov = (focal_vol / total_vol) if total_vol > 0 else 0.0
        comp_sov = (comp_vol / total_vol) if total_vol > 0 else 0.0
        
        return {
            "Brand_Current_SoV": float(brand_sov),
            "Competitor_Current_SoV": float(comp_sov),
        }

    def compute_engagement_and_archive(self):
        """计算当前帖子曝光权重并存档高价值帖子到中央记忆库。"""
        # 简化权重：角色权重
        role_weight = {"brand": 2.0, "competitor": 2.0, "consumer": 1.0, "agent": 1.0}
        top_list = []
        for post in self._posts_current:
            # 找到对应消息以获取sender_role
            pid = post.get("postId")
            sender_role = None
            for m in (self._message_buffer_step + self._message_buffer_current):
                if m.get("post_id") == pid:
                    sender_role = str(m.get("sender_role", "agent")).lower()
                    break
            rw = float(role_weight.get(sender_role or "agent", 1.0))
            total_weight = rw
            top_list.append({"postId": pid, "totalExposureWeight": total_weight})
        # 排序并选取Top-N
        top_list.sort(key=lambda x: x["totalExposureWeight"], reverse=True)
        feed_capacity = int(self.hyperparams.get("feed_capacity", 20))
        self._top_posts_by_weight = top_list[: min(len(top_list), feed_capacity)]
        # 新归档（避免重复）
        archived_ids = {p.get("postId") for p in self.memory_base}
        self.newly_archived_posts = []
        for item in self._top_posts_by_weight[:5]:  # 每步归档Top-5
            pid = item["postId"]
            if pid not in archived_ids:
                # 找到帖子对象
                for p in self._posts_current:
                    if p.get("postId") == pid:
                        self.memory_base.append(p)
                        self.newly_archived_posts.append(pid)
                        break

    def _compute_bri_nps(self, brand_id: Optional[str]) -> Tuple[float, float]:
        """根据消费者当前状态计算BRI与NPS。"""
        brand_id = brand_id or self.focal_brand_id
        consumers = self.agents_by_type.get("consumer", [])
        if not consumers:
            return 0.0, 0.0
        caps, vals, emos, advs = [], [], [], []
        for c in consumers:
            if brand_id is None:
                # 若未指定品牌，选消费者第一个品牌
                bids = list(c.cbbe_state.keys())
                if not bids:
                    continue
                bid = bids[0]
            else:
                bid = brand_id
            cs = c.cbbe_state.get(bid)
            is_ = c.intent_state.get(bid)
            if cs and is_:
                caps.append(float(cs.get("capability_belief", 0.0)))
                vals.append(float(cs.get("value_resonance", 0.0)))
                emos.append(float(cs.get("emotional_bond", 0.0)))
                advs.append(float(is_.get("advocacy_intent", 0.0)))
        if not caps:
            return 0.0, 0.0
        w1, w2, w3 = self.bri_weights
        # Eq. (12)-(13): BRI_t = 100 * (CB_t + VC_t + EC_t) / 3
        bri = float(w1 * np.mean(caps) + w2 * np.mean(vals) + w3 * np.mean(emos)) * 100.0
        
        # Consumer internals keep the tanh state in [-1, 1]. The
        # paper defines Advocacy Intent on [0, 1] and reports NPS on the
        # standard point scale, so convert at the published boundary.
        paper_advs = [max(0.0, min(1.0, (a + 1.0) / 2.0)) for a in advs]
        promoters = sum(1 for a in paper_advs if a > 0.7)
        detractors = sum(1 for a in paper_advs if a < 0.3)
        total = len(advs)
        p_rate = promoters / total if total > 0 else 0.0
        d_rate = detractors / total if total > 0 else 0.0
        nps = 100.0 * (p_rate - d_rate)
        
        return bri, float(nps)

    def record_strategic_action(self, agent_id: Any, triggered_llm: bool, llm_output: Optional[Dict[str, Any]] = None):
        """记录战略Agent的行动（用于JSONL汇总）。"""
        self._strategic_actions_current.append({
            "agentId": str(agent_id),
            "triggeredLlmDecision": bool(triggered_llm),
            "llmOutput": llm_output or None,
        })

    @staticmethod
    def public_text(message: Any) -> str:
        return public_text(message)

    def public_log_last_round(self) -> List[Dict[str, Any]]:
        """All public messages of the previous round (official posts and UGC).

        This is the observation source S_t for the strategic agents.  It is
        distinct from the one-shot delivery buffer, which is consumed when the
        messages are distributed.
        """
        return list(getattr(self, "_last_step_messages", []) or [])

    def _dominant_topic_snapshot(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据消息生成主导话题快照，包含粗略owner与情感，并标记是否为新话题。"""
        posts_texts = [public_text(m) for m in messages]

        # 估计owner：看品牌/竞品关键词命中数
        import re
        def count_hits(texts: List[str], kws: List[str]) -> int:
            if not texts or not kws:
                return 0
            pattern = re.compile(r"|".join([re.escape(k) for k in kws]), flags=re.IGNORECASE)
            return sum(1 for s in texts if s and pattern.search(s))

        brand_hits = count_hits(posts_texts, self.brand_keywords)
        comp_hits = count_hits(posts_texts, self.competitor_keywords)
        if brand_hits > comp_hits:
            owner = "brand"
        elif comp_hits > brand_hits:
            owner = "competitor"
        else:
            owner = "mixed" if (brand_hits + comp_hits) > 0 else (self._last_dominant_owner or "UGC_Community")

        is_new = (owner != self._last_dominant_owner)
        self._last_dominant_owner = owner

        # 粗略整体情感
        pos_words = ["好", "赞", "棒", "推荐", "喜欢", "love", "great", "excellent"]
        neg_words = ["差", "烂", "贵", "失望", "讨厌", "hate", "bad", "terrible", "垃圾"]
        pos_re = re.compile(r"|".join(pos_words), flags=re.IGNORECASE)
        neg_re = re.compile(r"|".join(neg_words), flags=re.IGNORECASE)
        pos = sum(1 for s in posts_texts if pos_re.search(s))
        neg = sum(1 for s in posts_texts if neg_re.search(s))
        sentiment = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")

        topic = ", ".join(self.topic_keywords) if self.topic_keywords else ""
        return {
            "topic": topic,
            "owner": owner,
            "is_new": bool(is_new),
            "sentiment": sentiment,
        }

    def _snapshot_consumer_states(self, brand_id: Optional[str]) -> List[Dict[str, Any]]:
        out = []
        for c in self.agents_by_type.get("consumer", []):
            # 选品牌
            bids = list(c.cbbe_state.keys())
            target = brand_id or self.focal_brand_id
            bid = target if target in bids else (bids[0] if bids else None)
            if bid is None:
                continue
            cs = c.cbbe_state.get(bid, {})
            is_ = c.intent_state.get(bid, {})
            
            # Ensure we get the specific stance for this brand if available
            current_stance = c.stance_by_brand.get(bid, c.current_stance_global)
            
            out.append({
                "agentId": c.unique_id,
                "static_personality": {"driver": c.core_driver, "initial_stance": c.current_stance_global},
                "dynamic_state_vector": {
                    "competenceBelief": float(cs.get("capability_belief", 0.0)),
                    "valueCongruence": float(cs.get("value_resonance", 0.0)),
                    "emotionalConnection": float(cs.get("emotional_bond", 0.0)),
                    "purchaseIntent": float(is_.get("purchase_intent", 0.0)),
                    "advocacyIntent": float(is_.get("advocacy_intent", 0.0)),
                    "current_stance": current_stance,
                }
            })
        return out

    def _write_jsonl_record(self, t: int, brand_id: Optional[str], source_messages: List[Dict[str, Any]]):
        """按规范生成一个时间步的JSONL记录并保存到内存。"""
        brand_id = brand_id or self.focal_brand_id
        bri, nps = self._compute_bri_nps(brand_id)
        sov = self._compute_sov_from_messages(source_messages)
        dominant = self._dominant_topic_snapshot(source_messages)
        
        # Manually compute polarization for JSONL to ensure consistency with metrics node
        # (or ideally fetch from metrics node if available, but here we re-compute)
        snapshot = self._snapshot_consumer_states(brand_id)
        stances: Dict[str, int] = {"supporter": 0, "neutral": 0, "opponent": 0}
        for s in snapshot:
            vec = s.get("dynamic_state_vector", {})
            # Use normalized stance counting
            raw_stance = str(vec.get("current_stance") or "neutral").lower()
            if "support" in raw_stance: stance = "supporter"
            elif "oppon" in raw_stance or "oppos" in raw_stance: stance = "opponent"
            elif "neut" in raw_stance: stance = "neutral"
            else: stance = "neutral"
            
            if stance in stances:
                stances[stance] += 1
            else:
                stances["neutral"] += 1
        
        n = max(1, len(snapshot))
        pi_s = stances["supporter"] / n
        pi_n = stances["neutral"] / n
        pi_o = stances["opponent"] / n
        
        # Esteban-Ray Polarization (K=1)
        # Sum = 2 * (pi_s*pi_n + 2*pi_s*pi_o + pi_n*pi_o)
        raw_sum = 2.0 * (pi_s * pi_n * 1.0 + pi_s * pi_o * 2.0 + pi_n * pi_o * 1.0)
        polarization_index = max(0.0, min(1.0, raw_sum))

        # Appendix D.4: Backfire Rate = share of rounds (t >= 1) in which the
        # evidence-consistency gate attenuated an official post (gamma < 1).
        backfire_rate = None
        if int(t) >= 1:
            self.metric_rounds += 1
            if any(
                str(p.get("authorRole", "")).lower() == "brand" and float(p.get("verification_factor", 1.0)) < 1.0
                for p in self._posts_current
            ):
                self.backfire_rounds += 1
            backfire_rate = self.backfire_rounds / self.metric_rounds if self.metric_rounds else None

        record = {
            "timestep": int(t),
            "market_metrics": {
                "BRI": float(bri),
                "NPS": float(nps),
                # Public JSONL follows Appendix C reporting units: SoV and
                # polarization are percentages, while BRI/NPS are points.
                "SoV": 100.0 * float(sov.get("Brand_Current_SoV", 0.0)),
                "PolarizationIndex": 100.0 * float(polarization_index),
                "dominantTopic": dominant,
                "BackfireRate": backfire_rate,
            },
            "strategic_actions": list(self._strategic_actions_current),
            "consumer_generated_content": list(self._posts_current),
            "platform_state": {
                "memoryBaseSize": len(self.memory_base),
                "newlyArchivedPosts": list(self.newly_archived_posts),
                "topPostsByWeight": list(self._top_posts_by_weight),
            },
            "consumer_states_snapshot": snapshot,
        }
        self.jsonl_records.append(record)

    def step(self):
        """Executes one step of the model."""
        # 1. Distribute messages from previous step (including strategic injections)
        self.distribute_information()
        
        # 2. Agents execute their step (process messages, update state, generate UGC)
        self.schedule.step()
        # Mesa 3 increments ``steps`` before invoking this user step.
        
        # 3. Compute engagement and archive top posts (for memory)
        self.compute_engagement_and_archive()
        
        # 4. Record this step to JSONL history
        brands = self.agents_by_type.get("brand", [])
        bid = self.focal_brand_id or (getattr(brands[0], "unique_id", None) if brands else None)
        self._write_jsonl_record(t=self.steps, brand_id=bid, source_messages=self._message_buffer_step)

        # Record history snapshot
        snapshot = {
            "step": self.steps,
            "phase": "post_step",
            "messages": [
                {
                    "sender_id": m.get("sender_id"),
                    "sender_role": m.get("sender_role"),
                    "message_type": m.get("message_type"),
                    "content": m.get("content"),
                    "via": m.get("via", "direct"),
                } for m in self._message_buffer_step
            ],
            "metrics": self._compute_sov_from_messages(self._message_buffer_step),
        }
        self.history.append(snapshot)
        
        # 5. Flip buffers for next step
        self._last_step_messages = list(self._message_buffer_step)
        self._last_step_posts = list(self._posts_current)
        self._message_buffer_prev = list(self._message_buffer_current)
        self._message_buffer_current = []
        self._message_buffer_step = []
        self._message_fingerprints_current = set()
        self._posts_current = []
        self._strategic_actions_current = []

        # Collect data
        if self.datacollector is not None:
            self.datacollector.collect(self)

    def start_benchmark(self, exam_text: str, brand_id: Optional[str] = None):
        """Seed t=0 and fix one focal brand for the entire trajectory."""
        brands = list(self.agents_by_type.get("brand", []) or [])
        # ``SimulationNode`` keeps competitors in the mixed brand registry for
        # SoV reporting.  A focal trajectory may only select an actual
        # brand-role agent, never a competitor-role agent.
        focal_candidates = [
            b for b in brands
            if str(getattr(b, "role", "brand")).lower() == "brand"
        ]
        valid_ids = [
            str(getattr(b, "unique_id", ""))
            for b in focal_candidates
            if getattr(b, "unique_id", None) is not None
        ]
        requested = str(brand_id) if brand_id is not None else None
        self.focal_brand_id = requested if requested in valid_ids else (valid_ids[0] if valid_ids else None)
        # t=0 广播（仅消费者）
        for c in self.agents_by_type.get("consumer", []):
            msg = {
                "sender_id": None,
                "sender_role": "platform",
                "message_type": "benchmark",
                "content": {"exam_text": str(exam_text), "brand_id": self.focal_brand_id}
            }
            try:
                c.receive_message(msg)
                # 立即处理以生成初始UGC
                c._process_messages()
                c._process_memory()
                # 如果可用，生成UGC
            except Exception:
                pass
        # 首次结算：互动聚合、记忆库处理、演化（简报/指标）
        
        self.compute_engagement_and_archive()
        # JSONL记录（t=0）
        self._write_jsonl_record(t=0, brand_id=self.focal_brand_id, source_messages=self._message_buffer_current)
        self._last_step_messages = list(self._message_buffer_current)
        self._last_step_posts = list(self._posts_current)
        # 翻转缓冲，准备t=1的分发
        self._message_buffer_prev = list(self._message_buffer_current)
        self._message_buffer_current = []
        self._message_buffer_step = []
        self._posts_current = []
        self._message_fingerprints_current = set()

    def distribute_information(self):
        """在 t+1 开始分发 t 的消息：强制推送官方，概率推送其他。"""
        if not self._message_buffer_prev:
            return
        # 准备全体代理
        all_agents: List[Any] = []
        seen_agent_ids = set()
        for lst in self.agents_by_type.values():
            for agent in lst:
                uid = getattr(agent, "unique_id", id(agent))
                if uid in seen_agent_ids:
                    continue
                seen_agent_ids.add(uid)
                all_agents.append(agent)

        # Update aggregate and topic-specific heat.  The aggregate value is a
        # fallback for messages without a topic label.
        new_activity = len(self._message_buffer_prev)
        self.global_heat = self.global_heat * self.heat_decay_gamma + new_activity
        for topic in list(self.topic_heat):
            self.topic_heat[topic] *= self.heat_decay_gamma
        for message in self._message_buffer_prev:
            for topic in self._message_topics(message):
                self.topic_heat[topic] = self.topic_heat.get(topic, 0.0) + 1.0
        
        # 统计分发
        distributed = []
        for m in self._message_buffer_prev:
            sender_id = m.get("sender_id")
            sender_role = str(m.get("sender_role", "agent")).lower()
            sender_agent = self._get_agent_by_id(sender_id)
            influence = float(getattr(sender_agent, "influence", 0.0) or 0.0)

            # 官方强制推送：品牌与竞品
            # Eq. (9) treats official brand posts as broadcasts. Competitor
            # and UGC messages remain subject to non-broadcast sampling.
            forced = sender_role == "brand"
            for agent in all_agents:
                if getattr(agent, "unique_id", None) == sender_id:
                    continue
                topics = self._message_topics(m)
                message_heat = max((self.topic_heat.get(topic, 0.0) for topic in topics), default=self.global_heat)
                # Eq. (9): official brand posts are broadcasts (P_push = 1); other
                # items are sampled by sender influence and topic heat.
                p_push = 1.0 if forced else self._compute_effective_push_p(influence=influence, heat=message_heat)
                via = "forced_push" if forced else "probabilistic_push"
                if getattr(agent, "role", "") == "consumer":
                    # Eq. (10): P_ret(m, u, t) = P_push(m, t) * s_u * a_{m,u}
                    p_ret = p_push * self._retention_susceptibility(agent) * self._stance_alignment(m, agent)
                    deliver = self.random.random() < p_ret
                else:
                    deliver = forced or self.random.random() < p_push

                if deliver:
                    payload = dict(m)
                    payload["via"] = via
                    try:
                        agent.receive_message(payload)
                        distributed.append({"to": agent.unique_id, "via": via})
                    except Exception:
                        pass

        self._enforce_consumer_feed_capacity()

        # 记录分发结果到历史快照（本步开头）
        self.history.append({
            "step": self.steps,
            "phase": "distribution",
            "source_messages_count": len(self._message_buffer_prev),
            "distributed_count": len(distributed),
        })

    def get_competitor_brief(self, competitor_agent) -> Dict[str, Any]:
        """面向竞争者Agent的简报：提供内部帖子、SoV与主导话题。"""
        # Observed corpus: the previous round's public log, text only.
        prev_msgs = self.public_log_last_round()
        posts_texts = [public_text(m) for m in prev_msgs]

        # 指标（基于上一轮）
        metrics = self._compute_sov_from_messages(prev_msgs)

        # 主导话题（粗略）：看命中计数与平均情感
        import re
        def owner_count(owner_keywords: List[str]) -> int:
            if not owner_keywords:
                return 0
            pattern = re.compile(r"|".join([re.escape(k) for k in owner_keywords]), flags=re.IGNORECASE)
            return sum(1 for s in posts_texts if s and pattern.search(s))

        brand_hits = owner_count(self.brand_keywords)
        comp_hits = owner_count(self.competitor_keywords)
        if brand_hits > comp_hits:
            owner = "brand"
        elif comp_hits > brand_hits:
            owner = "competitor"
        else:
            owner = "mixed"
        is_new = (owner != self._last_dominant_owner)
        self._last_dominant_owner = owner

        # 粗略整体情感（上一轮）
        pos_words = ["好", "赞", "棒", "推荐", "喜欢", "love", "great", "excellent"]
        neg_words = ["差", "烂", "贵", "失望", "讨厌", "hate", "bad", "terrible", "垃圾"]
        import re as _re
        pos_re = _re.compile(r"|".join(pos_words), flags=_re.IGNORECASE)
        neg_re = _re.compile(r"|".join(neg_words), flags=_re.IGNORECASE)
        pos = sum(1 for s in posts_texts if pos_re.search(s))
        neg = sum(1 for s in posts_texts if neg_re.search(s))
        sentiment = "正面" if pos > neg else ("负面" if neg > pos else "中性")

        return {
            "current_timestep": self.steps,
            "internal_posts": posts_texts,
            "your_internal_metrics_json": {
                "My_Current_SoV": metrics.get("Competitor_Current_SoV", 0.0),
            },
            "brand_agent_internal_metrics_json": {
                "Brand_Current_SoV": metrics.get("Brand_Current_SoV", 0.0),
            },
            "internal_dominant_topic_json": {
                "owner": owner,
                "sentiment": sentiment,
                "is_new": is_new,
                "topic": ", ".join(self.topic_keywords) if self.topic_keywords else "",
            },
            "internal_historical_posts_list": posts_texts[-50:],
        }
        
    def _create_social_network(self):
        """创建社交网络
        
        根据指定的网络类型和参数创建社交网络。
        
        Returns:
            networkx.Graph: 社交网络图
        """
        # 仅根据品牌与消费者数量创建网络（已移除KOL）
        total_agents = self.num_consumers + self.num_brands
        
        if total_agents <= 1:
            return nx.empty_graph(total_agents)
        if self.network_type == "small_world":
            # Watts--Strogatz requires 0 <= k < n and an even k.  Tiny smoke
            # runs commonly have one consumer plus one brand, so use a complete
            # graph where a valid ring lattice cannot be constructed.
            requested_k = int(self.network_params.get("k", 4))
            k = min(max(2, requested_k), total_agents - 1)
            if k % 2:
                k -= 1
            if k < 2:
                return nx.complete_graph(total_agents)
            return nx.watts_strogatz_graph(
                n=total_agents, k=k, p=float(self.network_params.get("p", 0.1)),
                seed=self.random.randint(0, 1000000)
            )
        elif self.network_type == "scale_free":
            m = min(max(1, int(self.network_params.get("m", 2))), total_agents - 1)
            if total_agents <= 2:
                return nx.complete_graph(total_agents)
            return nx.barabasi_albert_graph(
                n=total_agents, m=m, seed=self.random.randint(0, 1000000)
            )
        else:  # random
            # 随机网络
            return nx.erdos_renyi_graph(
                n=total_agents,
                p=self.network_params.get("p", 0.1),
                seed=self.random.randint(0, 1000000)
            )
    
    def _normalize_stance_value(self, stance_cn: str, stance_key: Optional[str]) -> Optional[str]:
        key = (stance_key or "").strip().lower()
        if key in {"supporter", "neutral", "opponent"}:
            return key
        synonyms = {
            "֧����": "supporter",
            "支持者": "supporter",
            "支持": "supporter",
            "supporter": "supporter",
            "������": "neutral",
            "中立": "neutral",
            "neutral": "neutral",
            "������": "opponent",
            "反对者": "opponent",
            "反对": "opponent",
            "opponent": "opponent",
        }
        label = (stance_cn or "").strip()
        if label in synonyms:
            return synonyms[label]
        for cn, mapped in synonyms.items():
            if cn and cn in label:
                return mapped
        return None

    def _create_agents(self):
        """构建模型中的品牌与消费者代理"""
        # 品牌
        brands = []
        for i in range(self.num_brands):
            brand_name = f"Brand_{i+1}"
            marketing_budget = 100 + self.random.randint(0, 100)
            product_quality = 0.5 + self.random.random() * 0.5

            brand = Brand(
                model=self,
                name=brand_name,
                marketing_budget=marketing_budget,
                product_quality=product_quality
            )

            brands.append(brand)
            self.schedule.add(brand)

        self.agents_by_type["brand"] = brands

        # 消费者
        consumers: List[Any] = []
        pop_consumers: List[Dict[str, Any]] = []
        try:
            pop_consumers = list((self._population_spec or {}).get("consumers", []))
        except Exception:
            pop_consumers = []

        if pop_consumers:
            for spec in pop_consumers:
                influence_susceptibility = self.random.random()
                loyalty = self.random.random()
                core_driver = spec.get("driver_key") or spec.get("driver_cn")
                initial_stance_cn = str(spec.get("initial_stance_cn", "")).strip()
                initial_stance = self._normalize_stance_value(initial_stance_cn, spec.get("initial_stance_key"))
                consumer = Consumer(
                    model=self,
                    influence_susceptibility=influence_susceptibility,
                    loyalty=loyalty,
                    core_driver=core_driver if isinstance(core_driver, str) else None,
                    initial_stance=initial_stance,
                    persona_seed=spec.get("persona_seed"),
                    persona_seed_offset=spec.get("persona_seed_offset"),
                    persona_label=spec.get("persona_label"),
                )
                try:
                    infl_score = float(spec.get("initial_influence_score", 0.0))
                    import math
                    consumer.influence = float(math.log(max(0.0, infl_score) + 1.0))
                except Exception:
                    pass
                consumers.append(consumer)
                self.schedule.add(consumer)
        else:
            for _ in range(self.num_consumers):
                influence_susceptibility = self.random.random()
                loyalty = self.random.random()
                consumer = Consumer(
                    model=self,
                    influence_susceptibility=influence_susceptibility,
                    loyalty=loyalty
                )
                consumers.append(consumer)
                self.schedule.add(consumer)

        self.agents_by_type["consumer"] = consumers

    # ====== 校准访问器 ======
    def get_calibration_beta(self, driver: str) -> Optional[List[List[float]]]:
        try:
            beta = self._calibration_beta.get(driver)
            if beta is None:
                return None
            # Ensure nested lists of floats
            return [[float(x) for x in row] for row in beta]
        except Exception:
            return None

    def get_calibration_w(self, driver: str, kind: str) -> Optional[List[float]]:
        try:
            d = self._calibration_w.get(driver)
            if not isinstance(d, dict):
                return None
            key = "purchase" if kind == "purchase" else "advocacy"
            vec = d.get(key)
            if vec is None:
                return None
            return [float(x) for x in vec]
        except Exception:
            return None
    
    def record_purchase(self, purchase_info: Dict):
        """记录购买行为
        
        Args:
            purchase_info: 购买信息
        """
        self.purchase_history.append(purchase_info)
        
        # 通知相关品牌有销售发生
        brand_id = purchase_info.get("brand_id")
        if brand_id is not None:
            for brand in self.agents_by_type.get("brand", []):
                if brand.unique_id == brand_id:
                    brand.record_sale()
                    break
    
    def _calculate_brand_awareness(self) -> Dict[str, float]:
        """计算品牌知名度
        
        计算所有消费者对所有品牌的平均知名度。
        
        Returns:
            Dict[str, float]: 每个品牌的平均知名度
        """
        consumers = self.agents_by_type.get("consumer", [])
        brands = self.agents_by_type.get("brand", [])
        
        if not consumers or not brands:
            return {}
        
        # 计算每个品牌的平均知名度
        brand_awareness = {}
        for brand in brands:
            total_awareness = sum(consumer.brand_awareness.get(brand.unique_id, 0) for consumer in consumers)
            avg_awareness = total_awareness / len(consumers) if consumers else 0
            brand_awareness[brand.name] = avg_awareness
        
        return brand_awareness
    
    def _calculate_consumer_sentiment(self) -> Dict[str, float]:
        """计算消费者情绪
        
        计算所有消费者对所有品牌的平均情绪值。
        
        Returns:
            Dict[str, float]: 每个品牌的平均消费者情绪
        """
        consumers = self.agents_by_type.get("consumer", [])
        brands = self.agents_by_type.get("brand", [])
        
        if not consumers or not brands:
            return {}
        
        # 计算每个品牌的平均情绪
        brand_sentiment = {}
        for brand in brands:
            total_sentiment = sum(consumer.brand_sentiment.get(brand.unique_id, 0) for consumer in consumers)
            avg_sentiment = total_sentiment / len(consumers) if consumers else 0
            brand_sentiment[brand.name] = avg_sentiment
        
        return brand_sentiment
    
    def _calculate_purchase_intention(self) -> Dict[str, float]:
        """计算购买意向
        
        计算所有消费者对所有品牌的平均购买意向。
        
        Returns:
            Dict[str, float]: 每个品牌的平均购买意向
        """
        consumers = self.agents_by_type.get("consumer", [])
        brands = self.agents_by_type.get("brand", [])
        
        if not consumers or not brands:
            return {}
        
        # 计算每个品牌的平均购买意向
        purchase_intention = {}
        for brand in brands:
            total_intention = sum(consumer.purchase_intention.get(brand.unique_id, 0) for consumer in consumers)
            avg_intention = total_intention / len(consumers) if consumers else 0
            purchase_intention[brand.name] = avg_intention
        
        return purchase_intention
    
    def _calculate_sales_volume(self) -> Dict[str, int]:
        """计算销售量
        
        计算每个品牌的销售量。
        
        Returns:
            Dict[str, int]: 每个品牌的销售量
        """
        brands = self.agents_by_type.get("brand", [])
        
        if not brands:
            return {}
        
        # 计算每个品牌的销售量
        sales_volume = {}
        for brand in brands:
            sales_volume[brand.name] = brand.sales
        
        return sales_volume
    
    def _calculate_brand_reputation(self) -> Dict[str, float]:
        """计算品牌声誉
        
        计算每个品牌的声誉。
        
        Returns:
            Dict[str, float]: 每个品牌的声誉
        """
        brands = self.agents_by_type.get("brand", [])
        
        if not brands:
            return {}
        
        # 计算每个品牌的声誉
        brand_reputation = {}
        for brand in brands:
            brand_reputation[brand.name] = brand.reputation
        
        return brand_reputation
    
    # 已移除：_calculate_kol_influence（KOL模块不再支持）
    


    def export_jsonl(self, filepath: str) -> None:
        """将当前内存中的 JSONL 记录落盘至指定文件路径。

        Args:
            filepath: 目标文件路径（.jsonl）。若父目录不存在将自动创建。
        """
        try:
            import json
            import os
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                for rec in self.jsonl_records:
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
        except Exception:
            # 记录导出失败但不打断仿真流程
            pass
