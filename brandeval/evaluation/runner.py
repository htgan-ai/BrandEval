from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from brandeval.pipeline.nodes.base_node import SandboxState
from brandeval.pipeline.orchestrator import Orchestrator
from brandeval.config.loaders import DRIVER_KEY_TO_CN, STANCE_KEY_TO_CN
def summary_stats(series: List[float]) -> Dict[str, Any]:
    try:
        n = len(series)
        if n == 0:
            return {"mean": 0.0, "var": 0.0, "min": None, "max": None, "count": 0}
        mean = sum(series) / n
        var = 0.0
        if n >= 2:
            var = sum((x - mean) ** 2 for x in series) / n
        return {
            "mean": float(mean),
            "var": float(var),
            "min": float(min(series)),
            "max": float(max(series)),
            "count": int(n),
        }
    except Exception:
        return {"mean": 0.0, "var": 0.0, "min": None, "max": None, "count": len(series)}


class EvaluationRunner:
    """Batch runner for reproducible evaluation experiments."""

    def __init__(self, orchestrator: Orchestrator, output_dir: Optional[str] = None, verbose: bool = False, slim_output: bool = False, log_fn: Optional[Callable[[str], None]] = None):
        self.orchestrator = orchestrator
        self.output_dir = Path(output_dir or "./output")
        self.verbose = bool(verbose)
        self.slim_output = bool(slim_output)
        self.log_fn = log_fn
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.log_fn:
            try:
                self.log_fn(message)
            except Exception:
                pass

    @staticmethod
    def _fmt_metric(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except Exception:
            return "n/a"

    def _extract_scalar(self, metrics: Dict[str, Any]) -> float:
        # Prefer market BRI; fallback to avg purchase intent
        try:
            market = metrics.get("market", {})
            if isinstance(market.get("BRI"), (int, float)):
                return float(market.get("BRI", 0.0))
        except Exception:
            pass
        try:
            micro = metrics.get("micro", {})
            if isinstance(micro.get("purchaseIntent_mean"), (int, float)):
                return float(micro.get("purchaseIntent_mean", 0.0))
        except Exception:
            pass
        return 0.0

    def _persona_distribution(self) -> Dict[str, Any]:
        driver_order = list(DRIVER_KEY_TO_CN.keys())
        stance_order = ["supporter", "neutral", "opponent"]
        model = getattr(self.orchestrator, "model", None)
        consumers = []
        if model is not None:
            try:
                consumers = list(getattr(model.agents_by_type, "get", lambda *_: [])("consumer", []))
            except Exception:
                consumers = list((model.agents_by_type or {}).get("consumer", []))
        total = len(consumers)
        counts: Dict[tuple[str, str], int] = {}
        for d in driver_order:
            for s in stance_order:
                counts[(d, s)] = 0
        for cons in consumers:
            driver_key = getattr(cons, "core_driver", None)
            stance_key = getattr(cons, "current_stance_global", None)
            driver_key = driver_key if driver_key in DRIVER_KEY_TO_CN else driver_order[0] if driver_order else "value_oriented"
            stance_key = stance_key if stance_key in stance_order else "neutral"
            counts[(driver_key, stance_key)] = counts.get((driver_key, stance_key), 0) + 1
        personas: List[Dict[str, Any]] = []
        for d in driver_order:
            for s in stance_order:
                cnt = counts.get((d, s), 0)
                ratio = round(cnt / total, 4) if total else 0.0
                personas.append({
                    "driver_key": d,
                    "driver_label": DRIVER_KEY_TO_CN.get(d, d),
                    "stance_key": s,
                    "stance_label": STANCE_KEY_TO_CN.get(s, s),
                    "count": cnt,
                    "ratio": ratio,
                })
        return {"total": total, "personas": personas}

    def _compute_dynamic_metrics(self) -> Dict[str, Any]:
        """Compute BrandEval-Dynamic metrics over the run.

        - Final BRI: last BRI from market_metrics
        - ARV: std dev of the per-episode min-max-normalized BRI trajectory x100
        - Final Polarization Index: computed from last consumer snapshot
        - CRS: recovery time from minimum BRI back to the first supplied value
        - Final SoV: last values from market_metrics
        """
        model = getattr(self.orchestrator, "model", None)
        records: List[Dict[str, Any]] = []
        if model is not None:
            records = list(getattr(model, "jsonl_records", []) or [])

        # Appendix C defines the trajectory over t=1,...,T. The simulator
        # also logs a t=0 initialization record, which is metadata rather
        # than a metric timestep and must not enter ARV/CRS.
        metric_records = [
            record for record in records
            if int(record.get("timestep", 1) or 0) > 0
        ]
        if not metric_records:
            metric_records = records

        # BRI series
        bri_series: List[float] = []
        final_sov = 0.0
        final_polarization = 0.0

        if metric_records:
            for r in metric_records:
                mm = r.get("market_metrics", {})
                bri_series.append(float(mm.get("BRI", 0.0)))
            last_mm = metric_records[-1].get("market_metrics", {})
            final_sov = float(last_mm.get("SoV", 0.0) or last_mm.get("Brand_Current_SoV", 0.0) or 0.0)

            # Round records already carry PolarizationIndex in percent.
            final_polarization = float(last_mm.get("PolarizationIndex", 0.0))
            polarization_from_snapshot = "PolarizationIndex" not in last_mm
            # Fallback only if MetricsNode failed to compute it (0.0 is a valid result if truly no polarization)
            # But given the bug, let's double check.
            # If market_metrics has it, use it. If it's 0.0, it might be real or bug.
            # Let's re-calculate from snapshot using robust stance normalization to be safe.
            snap = metric_records[-1].get("consumer_states_snapshot", []) if polarization_from_snapshot else []
            if snap:
                stances = {"supporter": 0, "neutral": 0, "opponent": 0}
                for s in snap:
                    vec = s.get("dynamic_state_vector", {})
                    # Use standard key first, then global fallback
                    raw_stance = vec.get("current_stance") or vec.get("current_stance_global") or "neutral"
                    raw_stance = str(raw_stance).lower()
                    
                    if "support" in raw_stance: stance = "supporter"
                    elif "oppon" in raw_stance or "oppos" in raw_stance: stance = "opponent"
                    elif "neut" in raw_stance: stance = "neutral"
                    else: stance = "neutral"
                    
                    if stance in stances:
                        stances[stance] += 1
                    else:
                        stances["neutral"] += 1
                
                n = max(1, len(snap))
                pi_S = stances["supporter"] / n
                pi_N = stances["neutral"] / n
                pi_O = stances["opponent"] / n
                
                # Esteban-Ray for discrete 3-point scale (alpha=1.0)
                # Max polarization (0.5 Supporter, 0.5 Opponent) => 1.0
                # Formula: P = K * sum(pi_i * pi_j * |y_i - y_j|)
                # K=1.0. 
                # Terms: 2 * (pi_S*pi_N*1 + pi_S*pi_O*2 + pi_N*pi_O*1)
                raw_sum = 2.0 * (pi_S * pi_N * 1.0 + pi_S * pi_O * 2.0 + pi_N * pi_O * 1.0)
                # Snapshot fallback is a ratio; report it in percent like the records.
                final_polarization = 100.0 * max(0.0, min(1.0, float(raw_sum)))

        final_bri = float(bri_series[-1]) if bri_series else 0.0
        # Appendix C Eq. (18)-(19): normalize each episode before computing
        # trajectory volatility so ARV is independent of absolute severity.
        rv = 0.0
        if len(bri_series) >= 2:
            lo = min(bri_series)
            hi = max(bri_series)
            scale = hi - lo + 1e-12
            normalized = [(x - lo) / scale for x in bri_series]
            mean_norm = sum(normalized) / len(normalized)
            var_norm = sum((x - mean_norm) ** 2 for x in normalized) / len(normalized)
            rv = 100.0 * math.sqrt(max(0.0, var_norm))

        # Final NPS
        final_nps = 0.0
        if records:
            last_mm = records[-1].get("market_metrics", {})
            final_nps = float(last_mm.get("NPS", 0.0))

        # Appendix C Eq. (20)-(21): recovery is measured against the episode
        # entry value BRI_1, with no epsilon or zero-baseline substitution.
        crs = 0
        if bri_series:
            min_val = min(bri_series)
            t_bottom = bri_series.index(min_val)
            t_recover = -1
            baseline = bri_series[0]
            for i in range(t_bottom + 1, len(bri_series)):
                if bri_series[i] >= baseline:
                    t_recover = i
                    break
            if t_recover != -1:
                crs = t_recover - t_bottom
            else:
                # (T + 1) - t_bottom in the paper's 1-indexed notation.
                crs = len(bri_series) - t_bottom


        return {
            "trajectory_points": len(bri_series),
            "includes_initialization": any(r.get("timestep") == 0 for r in records),
            "Final_BRI": final_bri,
            # Units follow the round records (Appendix C): BRI and NPS in
            # points, SoV and Polarization in percent.  Values are never
            # rescaled by magnitude.
            "units": {"Final_BRI": "points", "Final_NPS": "points", "Final_SoV": "percent", "Final_PolarizationIndex": "percent", "ARV": "points"},
            "Final_NPS": final_nps,
            "ARV": rv,
            "RV": rv,
            "Final_PolarizationIndex": final_polarization,
            "Final_SoV": final_sov,
            "CRS": crs,
        }

    def run(self, background_info: Dict[str, Any], steps: int = 3, tag: Optional[str] = None) -> Path:
        state = SandboxState(step=0, background_info=background_info)
        scalars: List[float] = []
        dialogues_all: List[Dict[str, Any]] = []

        turn_report: List[Dict[str, Any]] = []
        self._log(f"Runner starting (steps={steps})")
        for t in range(1, steps + 1):
            step_start = time.time()
            state.step = t
            self._log(f"[Runner] ===== Begin t={t} =====")
            state.logs.append(f"[Runner] ===== Begin t={t} =====")
            state = self.orchestrator.run_step(state)
            
            # Check for API Failures in state
            api_failures = 0
            # Check brand proposals for failure message
            for agent_name, proposal in state.agent_proposals.items():
                if isinstance(proposal, dict) and "rationale_summary" in proposal:
                    if proposal["rationale_summary"] == "API call failed, please check API":
                        api_failures += 1
                        self._log(f"[Runner] Critical API Failure detected in {agent_name} at t={t}")
            
            if api_failures > 0:
                # Threshold check: simple strict check for now (any failure triggers warning, >50% aborts?)
                # User said "massive failures" (累计出现大规模调用失败).
                # Let's track cumulative failures.
                if not hasattr(self, "_cumulative_api_failures"):
                    self._cumulative_api_failures = 0
                self._cumulative_api_failures += api_failures
                
                # Threshold: if > 5 cumulative failures
                if self._cumulative_api_failures >= 5:
                    self._log(f"[Runner] Aborting run due to excessive API failures ({self._cumulative_api_failures}).")
                    break

            elapsed = time.time() - step_start
            scalars.append(self._extract_scalar(state.metrics))
            state.logs.append(f"[Runner] ===== End t={t} =====")
            self._log(f"[Runner] ===== End t={t} (elapsed {elapsed:.1f}s) =====")
            try:
                mdl = getattr(self.orchestrator, "model", None)
                messages_out = []
                def _message_text(value):
                    if isinstance(value, dict):
                        nested = value.get("content")
                        if isinstance(nested, dict):
                            nested = nested.get("content") or nested.get("text")
                        return str(nested if nested is not None else value.get("text", ""))
                    return str(value or "")
                if mdl is not None:
                    h = getattr(mdl, "history", [])
                    if h:
                        last = h[-1]
                        for m in last.get("messages", []) or []:
                            messages_out.append({
                                "timestep": t,
                                "role": m.get("sender_role"),
                                "sender_id": m.get("sender_id"),
                                "content": _message_text(m.get("content")),
                            })
                    # 合并消费者UGC（来自 JSONL 记录）
                    # 注意：如果 model.history 已包含所有消息，则无需再次从 jsonl_records 添加。
                    # 这里添加去重逻辑，防止重复添加。
                    try:
                        recs = list(getattr(mdl, "jsonl_records", []) or [])
                        if recs:
                            last_rec = recs[-1]
                            posts = list(last_rec.get("consumer_generated_content", []) or [])
                            
                            # 构建已存在消息的指纹集合
                            existing_fingerprints = set()
                            for m in messages_out:
                                # 指纹：(role, sender_id, content)
                                # Ensure content is hashable (string)
                                content_str = _message_text(m.get("content"))
                                fp = (m.get("role"), m.get("sender_id"), content_str)
                                existing_fingerprints.add(fp)
                            
                            for p in posts:
                                aid = str(p.get("authorId") or "")
                                explicit_role = str(p.get("authorRole") or p.get("sender_role") or "").lower()
                                if explicit_role in {"brand", "competitor", "consumer"}:
                                    role = explicit_role
                                elif aid.startswith("consumer_"):
                                    role = "consumer"
                                elif aid.startswith("competitor_"):
                                    role = "competitor"
                                elif aid.startswith("brand_"):
                                    role = "brand"
                                else:
                                    role = ""
                                if role:
                                    content = _message_text(p.get("content"))
                                    content_str = content
                                    fp = (role, aid, content_str)
                                    if fp not in existing_fingerprints:
                                        messages_out.append({
                                            "timestep": t,
                                            "role": role,
                                            "sender_id": aid,
                                            "content": content,
                                        })
                                        existing_fingerprints.add(fp)
                    except Exception:
                        pass
                
                # Deduplicate identical role/sender/content triplets before adding to dialogues
                dedup_msgs = []
                seen = set()
                # Sort messages by role and sender_id for consistent output
                # Priority: Brand -> Competitor -> Consumer (or based on sender_id)
                # Helper to extract number from sender_id for natural sort (consumer_1 vs consumer_10)
                import re
                def natural_sort_key(msg):
                    role = msg.get("role", "")
                    sender = str(msg.get("sender_id", ""))
                    # Role priority: brand=0, competitor=1, consumer=2
                    role_prio = 0 if role == "brand" else 1 if role == "competitor" else 2
                    # Extract number
                    nums = re.findall(r'\d+', sender)
                    num = int(nums[0]) if nums else 0
                    # Sort by role, then by number, then by sender string (as tiebreaker)
                    return (role_prio, num, sender)

                # Sort messages_out before deduplication to ensure deterministic order
                messages_out.sort(key=natural_sort_key)

                for msg in messages_out:
                    content_str = str(msg.get("content", ""))
                    key = (
                        msg.get("role"),
                        msg.get("sender_id"),
                        content_str,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup_msgs.append(msg)
                
                dialogues_all.append({"timestep": t, "messages": dedup_msgs})

                brand_msgs = [m for m in dedup_msgs if m.get("role") == "brand"]
                competitor_msgs = [m for m in dedup_msgs if m.get("role") == "competitor"]
                consumer_msgs = [m for m in dedup_msgs if m.get("role") == "consumer"]
                for bm in brand_msgs:
                    snippet = str(bm.get("content", ""))[:200].replace("\n", " ")
                    self._log(f"[Runner] t={t} brand[{bm.get('sender_id')}]: {snippet}")
                for cm in competitor_msgs:
                    snippet = str(cm.get("content", ""))[:200].replace("\n", " ")
                    self._log(f"[Runner] t={t} competitor[{cm.get('sender_id')}]: {snippet}")
                for sample in consumer_msgs:
                    snippet = str(sample.get("content", ""))[:160].replace("\n", " ")
                    self._log(f"[Runner] t={t} consumer[{sample.get('sender_id')}]: {snippet}")

                def _round4_obj(obj):
                    if isinstance(obj, float):
                        return round(obj, 4)
                    if isinstance(obj, dict):
                        return {k: _round4_obj(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_round4_obj(x) for x in obj]
                    return obj
                metrics = _round4_obj(state.metrics or {})
                brand_spoke = any(m.get("role") == "brand" for m in dedup_msgs)
                competitor_spoke = any(m.get("role") == "competitor" for m in dedup_msgs)
                consumers_spoke = len([m for m in dedup_msgs if m.get("role") == "consumer"])
                turn_report.append({
                    "timestep": t,
                    "messages": dedup_msgs,
                    "counts": {
                        "consumers": consumers_spoke,
                        "brand_spoke": bool(brand_spoke),
                        "competitor_spoke": bool(competitor_spoke),
                    },
                    "metrics": metrics,
                })
            except Exception as e:
                self._log(f"[Runner] Error in logging step {t}: {e}")
                # Ensure we log an empty entry so the array index matches timestep if possible
                if not any(d.get("timestep") == t for d in dialogues_all):
                    dialogues_all.append({"timestep": t, "messages": [], "error": str(e)})
            if self.verbose:
                try:
                    hist = getattr(self.orchestrator, "model", None)
                    msg_list = []
                    if hist is not None:
                        h = getattr(hist, "history", [])
                        if h:
                            last = h[-1]
                            msg_list = last.get("messages", []) or []
                    self._log(f"[Verbose] t={t} messages={len(msg_list)}")
                    for m in msg_list:
                        role = str(m.get("sender_role"))
                        content = str(m.get("content"))
                        
                        # Format for log parsing
                        if role == "consumer":
                            sender_id = m.get("sender_id", "Consumer-?")
                            self._log(f"[{sender_id}] generated content: \"{content}\"")
                        else:
                            self._log(f"[{role}] {content}")
                            
                    self._log(f"[Verbose] t={t} metrics={json.dumps(state.metrics, ensure_ascii=False)}")
                except Exception as e:
                    self._log(f"[Verbose] Error logging messages: {e}")

            # Incremental Save
            try:
                self._save_results(scalars, dialogues_all)
            except Exception as e:
                self._log(f"[Runner] Warning: Failed to save incremental results at t={t}: {e}")

        self._save_results(scalars, dialogues_all)
        return self.output_dir

    def _save_results(self, scalars: List[float], dialogues_all: List[Dict[str, Any]]):
        stats = summary_stats(scalars)
        result_dir = self.output_dir
        result_dir.mkdir(exist_ok=True)
        dynamic_metrics = self._compute_dynamic_metrics()
        persona_stats = self._persona_distribution()
        
        def _round4(obj):
            if isinstance(obj, float):
                return round(obj, 4)
            if isinstance(obj, dict):
                return {k: _round4(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_round4(x) for x in obj]
            return obj
            
        rounded_scalars = [round(x, 4) for x in scalars]
        if "mean" in stats:
            stats["mean"] = round(float(stats.get("mean", 0.0)), 4)
        if "var" in stats:
            stats["var"] = round(float(stats.get("var", 0.0)), 4)
            
        # 每轮市场指标序列（不含发言）
        model = getattr(self.orchestrator, "model", None)
        round_metrics: List[Dict[str, Any]] = []
        try:
            recs = list(getattr(model, "jsonl_records", []) or [])
            for idx, r in enumerate(recs):
                rm = {
                    "timestep": int(r.get("timestep", idx)),
                    "market_metrics": _round4(r.get("market_metrics", {})),
                }
                round_metrics.append(rm)
        except Exception:
            pass
            
        metrics_help = {
            "round_metrics": "逐轮指标含 t=0 初始化记录；BRI/NPS 为 [-100,100] 点数，SoV/Polarization 为百分比。",
            "scalars": "按时间步提取的单一评分序列（默认优先BRI），用于波动与均值统计。",
            "stats": "scalars 序列的均值/方差/最值/计数概要。",
            "brandEval.dynamic": "审计版汇总：BRI/NPS 为点数；SoV/Polarization 为百分比；ARV 为归一化轨迹标准差 x100；RV 为 ARV 兼容别名。",
            "persona_distribution": "本轮消费者在15种人格（驱动×立场）上的数量与占比。",
        }
        
        (result_dir / "metrics_record.json").write_text(json.dumps({
            "round_metrics": round_metrics,
            "scalars": rounded_scalars,
            "stats": stats,
            "brandEval": {"dynamic": _round4(dynamic_metrics)},
            "persona_distribution": persona_stats,
            "metrics_help": metrics_help,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        
        (result_dir / "agents_all.json").write_text(json.dumps(dialogues_all, ensure_ascii=False, indent=2), encoding="utf-8")
