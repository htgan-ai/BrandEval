from __future__ import annotations

import hashlib
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any, Dict, Optional

from brandeval.agents.brand_agent import BrandAgent
from brandeval.agents.brand_agent_ablation import BrandAgentNoSR
from brandeval.agents.competitor_agent import CompetitorAgent
from brandeval.pipeline.nodes.base_node import BaseNode, SandboxState
from brandeval.gemini_data import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from brandeval.model import BrandEvalModel, public_text
from brandeval.pipeline.message_impact import MessageImpactRater


class SimulationNode(BaseNode):
    """t.2 Simulation execution within the digital sandbox.

    - Initializes/receives a BrandEvalModel and executes one time step.
    - Applies the `final_strategy` seed selection.
    - Captures basic outputs and model history snapshot.
    """

    name = "SimulationNode"

    def __init__(
        self,
        model: Optional[BrandEvalModel] = None,
        num_consumers: int = 200,
        num_brands: int = 2,
        network_type: str = "small_world",
        seed: Optional[int] = 7,
        population_spec: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        brand_agent_cfg: Optional[Dict[str, Any]] = None,
        background_context: Optional[str] = None,
        brand_policy: str = "sr",
    ):
        self.seed = int(seed if seed is not None else 7)
        if brand_policy not in {"sr", "no_sr"}:
            raise ValueError("brand_policy must be 'sr' (Strategic Rationale workflow) or 'no_sr' (single-action ablation)")
        self.brand_policy = brand_policy
        # Independent rater for the impact vector x_t of official messages.
        self._impact_rater = MessageImpactRater()
        self._rng = random.Random(self.seed)
        # A single worker is the reproducible default.  Parallel LLM calls can
        # still be enabled explicitly, but completion order then becomes part
        # of the trajectory and cannot be claimed bitwise deterministic.
        self._consumer_parallelism = int(os.environ.get("CONSUMER_LLM_CONCURRENCY", "1"))
        
        self.background_context = background_context

        # 若提供外部人口配置，则按需求随机抽样消费者与竞品数量
        sampled_spec = population_spec or None
        if population_spec and isinstance(population_spec.get("consumers"), list):
            try:
                templates = list(population_spec.get("consumers", []) or [])
                k = max(1, int(num_consumers))
                # 以有放回的方式从15种人格模板中扩展到目标数量
                pool: list = []
                if not templates:
                    # An explicitly empty population means "use the model's
                    # generated defaults"; do not index an empty template
                    # list while preparing the strategic agents.
                    sampled_spec = {"consumers": [], "competitors": list((population_spec.get("competitors") or []) or [])}
                    templates = []
                elif len(templates) >= k:
                    self._rng.shuffle(templates)
                    pool = templates[:k]
                else:
                    self._rng.shuffle(templates)
                    # 复制模板并生成唯一 agent_id
                    for i in range(k):
                        base = templates[i % max(1, len(templates))].copy()
                        base["agent_id"] = f"consumer_{i+1:03d}"
                        # 轻微扰动初始影响与立场（保证差异）
                        sc = float(base.get("initial_influence_score", 1.0) or 1.0)
                        # P5: 消费者影响力初始化噪声幅度
                        noise_delta = float((hyperparams or {}).get("initial_influence_noise_delta", 0.10))
                        # ϵ ~ Unif[1-delta, 1+delta]
                        # 0.9 + 0.2 * random() corresponds to delta=0.1
                        # New formula: (1.0 - noise_delta) + (2.0 * noise_delta * random())
                        base["initial_influence_score"] = round(sc * ((1.0 - noise_delta) + (2.0 * noise_delta * self._rng.random())), 3)
                        pool.append(base)
                comps = list((population_spec.get("competitors") or []) or [])
                if len(comps) > 2:
                    self._rng.shuffle(comps)
                    comps = comps[:2]
                if templates:
                    sampled_spec = {"consumers": pool, "competitors": comps}
                else:
                    sampled_spec["competitors"] = comps
            except Exception as e:
                print(f"[SimulationNode] population sampling failed: {e}")
                sampled_spec = population_spec
        self.model = model or BrandEvalModel(
            num_consumers=num_consumers if not sampled_spec else max(1, len(sampled_spec.get("consumers", [])) or num_consumers),
            num_brands=num_brands,
            network_type=network_type,
            seed=self.seed,
            population_spec=sampled_spec,
            calibration=calibration,
            hyperparams=hyperparams,
        )
        self._population_spec = sampled_spec
        # t=0 基准初始化守护
        self._initialized_benchmark: bool = False
        # t.1 战略Agent配置
        self._brand_agent_cfg: Dict[str, Any] = dict(brand_agent_cfg or {})
        self._brand_agent: Optional[BrandAgent] = None
        self._competitors: list[CompetitorAgent] = []
        try:
            self._competitor_specs = list((self._population_spec or {}).get("competitors", []))
        except Exception:
            self._competitor_specs = []
        self._last_strategy: Dict[str, Any] = {}
        supplied_evidence = (hyperparams or {}).get("evidence_store", {})
        self.evidence_store: Dict[str, Any] = (
            dict(supplied_evidence)
            if isinstance(supplied_evidence, dict)
            else {str(item): {} for item in supplied_evidence}
            if isinstance(supplied_evidence, (list, tuple, set))
            else {}
        )

    def _ensure_brand_agent(self, model_name: Optional[str]) -> None:
        if self._brand_agent is None:
            cfg = self._brand_agent_cfg
            identity = cfg.get("identity") if isinstance(cfg.get("identity"), dict) else None
            try:
                goal_bri = float(cfg.get("goal_bri_floor", 60.0))
            except Exception:
                goal_bri = 60.0
            try:
                goal_nps = float(cfg.get("goal_nps_floor", 0.2))
            except Exception:
                goal_nps = 0.2

            name = str(cfg.get("agent_id") or "BrandAgent")
            brand_name = str(cfg.get("brand_name") or "My Brand")
            product_name = str(cfg.get("product_name") or "My Product")
            agent_cls = BrandAgentNoSR if self.brand_policy == "no_sr" else BrandAgent
            self._brand_agent = agent_cls(
                model=self.model,
                llm=None,
                identity=identity,
                name=name,
                brand_name=brand_name,
                product_name=product_name,
                goal_bri_floor=goal_bri,
                goal_nps_floor=goal_nps,
            )
        if model_name and hasattr(self._brand_agent, "use_gemini_llm"):
            try:
                self._brand_agent.use_gemini_llm(model_name=model_name)
                # Only infer identity if using defaults (meaning manual injection wasn't done or was default)
                # But user explicitly requested manual input and "not recommended auto extraction".
                # So if we have manual names (which we pass now), we skip inference.
                # A simple check: if brand_name is not "My Brand" (default in BrandAgent __init__ if not passed),
                # we assume it was manually set.
                # However, in _ensure_brand_agent above, we default to "My Brand" if cfg is missing it.
                # Let's check if self._brand_agent.brand_name is "My Brand".
                if self._brand_agent.brand_name == "My Brand" and self.background_context and hasattr(self._brand_agent, "infer_identity"):
                     self._brand_agent.infer_identity(self.background_context)
            except Exception:
                pass
        self._sync_strategic_agents()

    def _ensure_competitor_agents(self, model_name: Optional[str]) -> None:
        if not self._competitors:
            specs = self._competitor_specs or [{"agent_id": "competitor_01", "persona": "defender"}]
            for idx, spec in enumerate(specs, start=1):
                persona = str(spec.get("persona") or "defender")
                name = str(spec.get("agent_id") or f"competitor_{idx:02d}")
                try:
                    comp = CompetitorAgent(model=self.model, persona=persona, name=name)
                except Exception as e:
                    continue

                try:
                    my_kws = spec.get("keywords", [])
                    brand_kws = spec.get("brand_keywords", [])
                    if isinstance(my_kws, list) or isinstance(brand_kws, list):
                        comp.register_keywords(list(my_kws or []), list(brand_kws or []))
                except Exception:
                    pass
                
                if model_name and hasattr(comp, "use_gemini_llm"):
                    try:
                        comp.use_gemini_llm(model_name=model_name)
                    except Exception:
                        pass
                
                try:
                    self.model.schedule.add(comp)
                    self._competitors.append(comp)
                except Exception as e:
                    pass
        if model_name:
            for comp in self._competitors:
                try:
                    comp.use_gemini_llm(model_name=model_name)
                except Exception:
                    continue
        self._sync_strategic_agents()
    
    def _sync_strategic_agents(self) -> None:
        """Sync strategic brand/competitor agents to the model's main agent registry."""
        # Ensure they are in the model's agents_by_type so SoV metrics work correctly.
        # model.py assumes agents_by_type["brand"] contains [focal_brand, competitor_1, ...]
        
        new_list = []
        # 1. Focal Brand
        if self._brand_agent:
            new_list.append(self._brand_agent)
        
        # 2. Competitors (treated as secondary brands in the registry for metric calculation)
        if self._competitors:
            new_list.extend(self._competitors)
            
        # 3. If we have strategic agents, overwrite the dummy brands.
        #    If not, keep dummy brands (fallback).
        if new_list:
            self.model.agents_by_type["brand"] = new_list
        
        # Also sync competitors to their own type key
        if self._competitors:
            self.model.agents_by_type["competitor"] = list(self._competitors)
        
        # 4. Force update consumers to recognize these new strategic agents
        # (Because consumers were initialized with dummy brands, they don't know the new IDs)
        consumers = self.model.agents_by_type.get("consumer", [])
        all_strategic = []
        if self._brand_agent:
            all_strategic.append(self._brand_agent)
        if self._competitors:
            all_strategic.extend(self._competitors)
            
        if consumers and all_strategic:
            for cons in consumers:
                if hasattr(cons, "register_brand"):
                    for agent in all_strategic:
                        aid = getattr(agent, "unique_id", None)
                        if aid:
                            cons.register_brand(str(aid))
                # Remove placeholder states created before the strategic
                # agents were attached.  Otherwise t=0 can read a stale
                # dummy brand and the focal trajectory is not well-defined.
                valid_ids = {str(getattr(agent, "unique_id", "")) for agent in all_strategic}
                for mapping_name in (
                    "brand_awareness", "brand_sentiment", "purchase_intention",
                    "cbbe_state", "intent_state", "stance_by_brand",
                ):
                    mapping = getattr(cons, mapping_name, None)
                    if isinstance(mapping, dict):
                        for bid in list(mapping):
                            if str(bid) not in valid_ids:
                                mapping.pop(bid, None)

    def _configure_consumer_generation(
        self,
        temp_min: float,
        temp_max: float,
        consumer_model_name: Optional[str],
    ) -> None:
        try:
            t_min = float(temp_min)
            t_max = float(temp_max)
        except Exception:
            t_min, t_max = 0.35, 0.85
        if t_min > t_max:
            t_min, t_max = t_max, t_min
        consumers = list(self.model.agents_by_type.get("consumer", []) or [])
        for cons in consumers:
            model_to_use = consumer_model_name or DEFAULT_LLM_MODEL
            try:
                setattr(cons, "llm_model_name", model_to_use)
            except Exception:
                pass
            try:
                seed_hint = getattr(cons, "persona_seed", self.seed)
                seed_offset = getattr(cons, "persona_seed_offset", 0)
                material = f"{seed_hint}-{seed_offset}-{getattr(self.model, 'steps', 0)}"
                digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
                derived_seed = int(digest[:8], 16)
                rng = random.Random(derived_seed)
                if hasattr(cons, 'use_gemini_llm'):
                    cons.use_gemini_llm(model_to_use)
                cons.llm_temp_min = t_min
                cons.llm_temp_max = t_max
                cons.llm_temp_jitter = rng.uniform(0.0, 0.2)
            except Exception:
                if hasattr(cons, 'use_gemini_llm'):
                    try:
                        cons.use_gemini_llm(model_to_use)
                    except Exception:
                        pass
                cons.llm_temp_min = t_min
                cons.llm_temp_max = t_max
                cons.llm_temp_jitter = 0.0

    def enable_agent_llms(
        self,
        model_name: Optional[str] = None,
        temp_min: float = 0.35,
        temp_max: float = 0.85,
        brand_model_name: Optional[str] = None,
        competitor_model_name: Optional[str] = None,
        consumer_model_name: Optional[str] = None,
    ) -> "SimulationNode":
        base_model_name = model_name or consumer_model_name or brand_model_name
        self._impact_rater = MessageImpactRater.from_model_name(base_model_name)
        self._ensure_brand_agent(brand_model_name or base_model_name)
        self._ensure_competitor_agents(competitor_model_name or brand_model_name or base_model_name)
        self._configure_consumer_generation(
            temp_min=temp_min,
            temp_max=temp_max,
            consumer_model_name=consumer_model_name or base_model_name,
        )
        return self

    def _apply_strategy(self, strategy: Optional[Dict[str, Any]]) -> None:
        if isinstance(strategy, dict):
            self._last_strategy = dict(strategy)
        else:
            self._last_strategy = {}

    # ====== t=0：研究者播种 + 平台广播 ======
    def _ensure_benchmark_initialized(self, background: Dict[str, Any]) -> None:
        if self._initialized_benchmark:
            return
        try:
            exam_text = str(background.get("exam_text") or background.get("dominant_topic") or "Benchmark")
            brand_id = background.get("brand_id")
            # 注册话题与关键词（若提供）
            topic = str(background.get("dominant_topic") or "").strip()
            if topic:
                try:
                    self.model.register_topic_keywords([topic])
                except Exception:
                    pass
            bk = background.get("brand_keywords") or []
            ck = background.get("competitor_keywords") or []
            # 若未提供关键词，使用模型中的品牌/竞品名称作为默认关键词
            if not bk:
                try:
                    bk = [getattr(b, "name", "") for b in self.model.agents_by_type.get("brand", []) if getattr(b, "name", "")]
                except Exception:
                    bk = bk
            if not ck:
                try:
                    ck = [getattr(c, "name", "") for c in self._competitors if getattr(c, "name", "")]
                except Exception:
                    ck = ck
            try:
                self.model.register_brand_competitor_keywords(list(bk), list(ck))
            except Exception:
                pass
            # 平台执行 t=0 初始化与结算
            self.model.start_benchmark(exam_text=exam_text, brand_id=brand_id)
            self._initialized_benchmark = True
        except Exception:
            # 保持健壮性，不阻断后续步骤
            self._initialized_benchmark = True

    # ====== t.1：战略决策阶段（品牌/竞品） ======
    def _build_brand_brief(self, step: int) -> Dict[str, Any]:
        # Observation S_t: the previous round's public log, projected to the
        # visible text and platform aggregates.  Latent states, impact vectors,
        # prompts, and evaluation metrics (BRI, NPS, Pol., ARV, CRS) are never
        # part of the brief.
        prev_msgs = list(getattr(self.model, "_last_step_messages", []) or [])
        try:
            sov = self.model._compute_sov_from_messages(prev_msgs)
        except Exception:
            sov = {}
        # 主导话题（上一轮）
        try:
            dom = self.model._dominant_topic_snapshot(prev_msgs)
            dominant_topic_string = dom.get("topic", "")
        except Exception:
            dominant_topic_string = ""
        competitor_last_move = ""
        for message in reversed(prev_msgs):
            if str(message.get("sender_role", "")).lower() != "competitor":
                continue
            competitor_last_move = public_text(message).strip()
            if competitor_last_move:
                break
        # 负面帖子摘取（上一轮）
        neg_words = ["差", "烂", "贵", "失望", "讨厌", "hate", "bad", "terrible", "垃圾"]
        key_negative_posts: list[str] = []
        try:
            import re
            neg_re = re.compile(r"|".join(neg_words), flags=re.IGNORECASE)
            for m in prev_msgs:
                txt = public_text(m)
                if txt and neg_re.search(txt):
                    key_negative_posts.append(txt)
            key_negative_posts = key_negative_posts[:10]
        except Exception:
            key_negative_posts = []
        # 汇总简报
        metrics_json_object = {
            "Brand_Current_SoV": float(sov.get("Brand_Current_SoV", 0.0) or 0.0),
            "Competitor_Current_SoV": float(sov.get("Competitor_Current_SoV", 0.0) or 0.0),
        }
        return {
            "current_timestep": int(step),
            "dominant_topic": dominant_topic_string,
            "key_negative_posts": key_negative_posts,
            "metrics": metrics_json_object,
            "competitor_last_move": competitor_last_move,
        }

    def _append_official_post_to_prev(self, sender_role: str, text: str, brand_id: Optional[str] = None,
                                      info_impact: Optional[Dict[str, float]] = None, sender_id: Optional[str] = None,
                                      verification_factor: float = 1.0) -> None:
        """将官方帖子直接注入上一轮缓冲，以便在当前步进行分发。并同步记录帖子列表。"""
        try:
            # 生成postId
            post_id = f"post_{self.model._next_post_id}"
            self.model._next_post_id += 1
        except Exception:
            post_id = None

        # 注入上一轮缓冲（用于分发）
        msg = {
            "sender_id": sender_id,
            "sender_role": sender_role,
            "message_type": "official_post",
            "brand_id": brand_id,
            "content": {
                "content": text,
                "brand_id": brand_id,
                "info_impact": info_impact or {},
                "verification_factor": verification_factor
            },
            "post_id": post_id,
            "via": "forced_push",
        }
        try:
            self.model._message_buffer_prev.append(msg)
            # The previous buffer controls delivery; the step buffer controls
            # the metrics/JSONL record for this same round.  Keep one copy in
            # each purpose-specific buffer, without re-ingesting it through
            # record_message (which would create a second current message).
            self.model._message_buffer_step.append(msg)
        except Exception:
            pass

        # 记录规范化帖子（用于JSONL）
        try:
            self.model._posts_current.append({
                "postId": post_id or "",
                "authorId": sender_id,
                "authorRole": sender_role,
                "authorPersonality": None,
                "contentType": "new_post",
                "content": text,
                "inReplyTo": None,
                "topicTags": [],
                "brand_id": brand_id,
                "info_impact": info_impact or {},
                "verification_factor": verification_factor,
            })
        except Exception:
            pass

        # The previous buffer is the single source for the next distribution.
        # Do not also record the same post into the current buffer: doing both
        # caused one strategic action to be delivered and counted twice.

    def _t1_strategic_phase(self, state: SandboxState) -> None:
        """t.1: the focal agent publishes one action per round; then competitors act."""
        if self._brand_agent and state.step >= 1:
            try:
                if hasattr(self.model, "get_brand_brief"):
                    brief = self.model.get_brand_brief(brand_agent=self._brand_agent) or {}
                else:
                    brief = self._build_brand_brief(step=state.step)

                self.model._defer_agent_broadcast = True
                try:
                    action = self._brand_agent.strategic_decision(brief)
                finally:
                    self.model._defer_agent_broadcast = False
                try:
                    self.model.record_strategic_action(agent_id=self._brand_agent.name, triggered_llm=bool(self._brand_agent.llm), llm_output=action.get("final_action"))
                except Exception:
                    pass
                final_action = action.get("final_action", {}) if isinstance(action, dict) else {}
                brand_text = str(final_action.get("final_action_content") or "").strip()
                if not brand_text or "API call failed" in brand_text:
                    # Deterministic offline fallback keeps the sandbox executable
                    # when the endpoint is disabled or unavailable.
                    try:
                        final_action = self._brand_agent._heuristic_action(
                            str(brief.get("dominant_topic", "")),
                            list(brief.get("key_negative_posts", []) or []),
                        )
                        brand_text = str(final_action.get("final_action_content", "")).strip()
                    except Exception:
                        final_action = {}
                        brand_text = "品牌已发布说明并保持公开透明，欢迎提出问题。"
                if brand_text and "No significant action taken" not in brand_text:
                    # The published text is normalized and truncated first; the
                    # rater sees exactly that string (never the agent's own predicted_impact).
                    brand_text = brand_text.strip()[:5000]
                    info_vec = self._impact_rater.rate(brand_text, "brand")
                    # Appendix C.1.2: the evidence-consistency gate checks claimed
                    # evidence identifiers against the evidence store E(t).
                    claimed = final_action.get("claimed_evidence", []) if isinstance(final_action, dict) else []
                    v_factor = self._verify_action_claims(claimed)
                    bid = getattr(self._brand_agent, "unique_id", "brand_0")
                    self._append_official_post_to_prev(sender_role="brand", text=brand_text, brand_id=bid, info_impact=info_vec, sender_id=bid, verification_factor=v_factor)
            except Exception:
                pass

        self._competitor_phase()

    def _verify_action_claims(self, claimed_evidence: list) -> float:
        """Evidence-consistency gate γ(t) ∈ [0, 1] of Appendix C.1.2.

        γ is the share of claimed evidence identifiers that exist in the
        evidence store E(t); 1.0 when no evidence is claimed.  Membership, not
        lexical resemblance, is the criterion.
        """
        if not claimed_evidence:
            return 1.0
        if isinstance(claimed_evidence, str):
            claimed_evidence = [claimed_evidence]
        if not isinstance(claimed_evidence, (list, tuple, set)):
            return 0.0
        identifiers = [str(item).strip() for item in claimed_evidence if str(item).strip()]
        if not identifiers:
            return 1.0
        supported = sum(1 for identifier in identifiers if identifier in self.evidence_store)
        return supported / len(identifiers)

    def _competitor_phase(self) -> None:
        """Generate competitor actions in the same strategic phase as the brand.

        The previous implementation was indented after ``_verify_action_claims``
        returned, so ``competitor_pressure_mu`` was never evaluated.  Keeping
        this phase explicit also makes the pressure parameter testable.
        """
        if not self._competitors:
            return
        try:
            mu = max(0.0, float((self.model.hyperparams if self.model else {}).get("competitor_pressure_mu", 1.0)))
        except Exception:
            mu = 1.0
        for comp in self._competitors:
            count = 0
            if self._rng.random() < min(1.0, mu):
                count = 1
                if mu > 1.0 and self._rng.random() < min(1.0, mu - 1.0):
                    count = 2
            for _ in range(count):
                try:
                    brief = self.model.get_competitor_brief(comp) if hasattr(self.model, "get_competitor_brief") else {}
                except Exception:
                    brief = {}
                try:
                    self.model._defer_agent_broadcast = True
                    try:
                        action = comp.strategic_decision(brief if isinstance(brief, dict) else {})
                    finally:
                        self.model._defer_agent_broadcast = False
                except Exception:
                    action = {}
                try:
                    self.model.record_strategic_action(
                        agent_id=getattr(comp, "name", "Competitor"),
                        triggered_llm=bool(getattr(comp, "llm", None)),
                        llm_output=(action or {}).get("final_action"),
                    )
                except Exception:
                    pass
                payload = action.get("final_action", {}) if isinstance(action, dict) else {}
                text = str(payload.get("final_action_content") or "").strip()
                if not text or "API call failed" in text:
                    try:
                        text = str(comp._heuristic_action(brief if isinstance(brief, dict) else {}).get("final_action_content", "")).strip()
                    except Exception:
                        text = "竞争对手发布声明以保持市场声量。"
                if text:
                    cid = getattr(comp, "unique_id", "competitor_dummy")
                    text = text.strip()[:5000]
                    self._append_official_post_to_prev(
                        sender_role="competitor", text=text, brand_id=cid,
                        info_impact=self._impact_rater.rate(text, "competitor"),
                        sender_id=cid,
                    )

    def run(self, state: SandboxState, **kwargs) -> SandboxState:
        # t=0 初始化（仅一次）
        self._ensure_benchmark_initialized(background=state.background_info or {})

        # Clear step buffer at the beginning of the simulation step
        if hasattr(self.model, "_message_buffer_step"):
            self.model._message_buffer_step = []

        step_num = int(getattr(state, "step", 1) or 1)
        if step_num >= 1:
            self._t1_strategic_phase(state)

        # 应用聚合策略（占位钩子）
        self._apply_strategy(state.final_strategy)

        try:
            try:
                brands = self.model.agents_by_type.get("brand", [])
                preferred_bid = getattr(self.model, "focal_brand_id", None) or (getattr(brands[0], "unique_id", None) if brands else None)
            except Exception:
                preferred_bid = None
            # 为用户构造上一轮官方/竞品摘要，作为本轮生成的语境
            prev_msgs = list(getattr(self.model, "_message_buffer_prev", []) or [])
            
            # Deliver the previous buffer exactly once before UGC generation.
            # ``model.step`` would otherwise distribute the same messages again.
            if prev_msgs and hasattr(self.model, "distribute_information"):
                self.model.distribute_information()
                self.model._message_buffer_prev = []
                for c in self.model.agents_by_type.get("consumer", []):
                    if hasattr(c, "_process_messages"):
                        c._process_messages()
            
            try:
                official_texts = []
                for m in prev_msgs:
                    if str(m.get("message_type")) == "official_post":
                        txt = m.get("content") if isinstance(m.get("content"), str) else m.get("content", {}).get("content")
                        if txt:
                            official_texts.append(str(txt))
                seed_text = (" \n".join(official_texts))[:500] if official_texts else ""
            except Exception:
                seed_text = ""
            consumers = list(self.model.agents_by_type.get("consumer", []) or [])
            competitors = self.model.agents_by_type.get("competitor", [])
            comp_id = getattr(competitors[0], "unique_id", None) if competitors else None
            
            tasks = {}
            if consumers:
                # Robust execution with timeout protection
                total_tasks = 0
                completed_tasks = 0
                
                with ThreadPoolExecutor(max_workers=max(1, self._consumer_parallelism)) as pool:
                    for cons in consumers:
                        try:
                            setattr(cons, "strict_one_per_step", True)
                        except Exception:
                            pass
                        uid = getattr(cons, "unique_id", "consumer")
                        
                        # Determine target brand for UGC to balance SoV
                        # 30% chance to discuss competitor if available
                        target_bid = preferred_bid
                        if comp_id and self._rng.random() < 0.3:
                            target_bid = comp_id
                        
                        # Ensure we don't pass None as target_bid
                        if not target_bid and brands:
                            target_bid = getattr(brands[0], "unique_id", "brand_1")
                        if not target_bid:
                            target_bid = "brand_1"
                        
                        # FORCE GENERATION:
                        # We specifically use force_generate_ugc_for_brand if available, or fallback to standard method
                        if hasattr(cons, "force_generate_ugc_for_brand"):
                            fut = pool.submit(cons.force_generate_ugc_for_brand, target_bid)
                        elif hasattr(cons, "_generate_ugc_for_brand"):
                            fut = pool.submit(cons._generate_ugc_for_brand, target_bid, seed_text=seed_text)
                        elif hasattr(cons, "force_generate_ugc_one"):
                            fut = pool.submit(cons.force_generate_ugc_one, preferred_brand_id=target_bid)
                        else:
                            continue
                        tasks[fut] = uid
                        total_tasks += 1
                    
                    # Wait for results with timeout to avoid hanging forever
                    try:
                        # Allow up to 180 seconds of silence before timing out the batch
                        for fut in as_completed(tasks, timeout=180):
                            try:
                                fut.result()
                                completed_tasks += 1
                            except Exception as e:
                                uid = tasks.get(fut, "unknown")
                    except TimeoutError:
                        pass
                    try:
                        # Increase timeout to 1800s (30 minutes) to handle 100 consumers
                        for fut in as_completed(tasks, timeout=1800):
                            cid = tasks[fut]
                            try:
                                fut.result()
                            except Exception as exc:
                                print(f"[SimulationNode] consumer {cid} generation failed: {exc}")
                                continue
                    except TimeoutError:
                        print('[SimulationNode] consumer batch timed out after 1800s; cancelling pending')
                        for fut in tasks:
                            if not fut.done():
                                fut.cancel()
            
            # Set flag to prevent double execution in model.step() -> consumer.step()
            self.model.manual_consumer_execution = True
            self.model.manual_strategic_execution = True
            try:
                # 执行当步调度
                self.model.step()
            finally:
                # Clear flag
                self.model.manual_consumer_execution = False
                self.model.manual_strategic_execution = False
        except Exception as e:
            print(f"[SimulationNode] CRITICAL ERROR in parallel execution: {e}")
            import traceback
            traceback.print_exc()
            
            # 若缺少调度器或出错，回退为简单串行循环（强制生成）
            self.model.manual_consumer_execution = False # Ensure flag is clear
            self.model.manual_strategic_execution = False
            print("[SimulationNode] Falling back to serial execution (FORCE generation)...")
            
            consumers = list(self.model.agents_by_type.get("consumer", []) or [])
            for cons in consumers:
                try:
                    # Force generation for everyone in serial mode
                    if hasattr(cons, "force_generate_ugc_one"):
                        cons.force_generate_ugc_one()
                    elif hasattr(cons, "_generate_ugc_for_brand"):
                         # Try to find a brand
                         bids = list(cons.intent_state.keys())
                         bid = bids[0] if bids else "brand_1"
                         cons._generate_ugc_for_brand(bid)
                    else:
                        cons.step()
                except Exception as ex:
                    print(f"[SimulationNode] Serial fallback failed for {getattr(cons, 'unique_id', '?')}: {ex}")

            # Run other agents
            for k, agents in getattr(self.model, "agents_by_type", {}).items():
                if k == "consumer": continue
                for agent in agents:
                    try:
                        agent.step()
                    except Exception:
                        pass

        # 收集输出快照
        snapshot = {
            "t": self.model.steps,
            "last_history": self.model.history[-1] if getattr(self.model, "history", []) else {},
        }
        state.simulation_outputs = snapshot
        state.logs.append(f"[SimulationNode] t={state.step} executed; model.t={self.model.steps}")
        return state
