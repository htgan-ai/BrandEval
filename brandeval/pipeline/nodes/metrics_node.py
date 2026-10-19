from __future__ import annotations

from typing import Any, Dict, List
import networkx as nx

from brandeval.pipeline.nodes.base_node import BaseNode, SandboxState


def calculate_brand_metrics(model) -> Dict[str, Any]:
    """最小品牌指标：每品牌的销量、营收、声誉。"""
    brands = model.agents_by_type.get("brand", [])
    metrics = {"sales": {}, "revenue": {}, "reputation": {}}
    for b in brands:
        name = getattr(b, "name", str(getattr(b, "unique_id", "brand")))
        metrics["sales"][name] = int(getattr(b, "sales", 0))
        metrics["revenue"][name] = float(getattr(b, "revenue", 0.0))
        metrics["reputation"][name] = float(getattr(b, "reputation", 0.0))
    return metrics


def calculate_consumer_metrics(model) -> Dict[str, Any]:
    """最小消费者指标：平均购买意向与平均情绪（按全体）。"""
    consumers = model.agents_by_type.get("consumer", [])
    avg_pi = 0.0
    avg_sentiment = 0.0
    if consumers:
        pis = []
        sentiments = []
        for c in consumers:
            sentiments.append(float(getattr(c, "sentiment", 0.0)))
            pi_vals = list(getattr(c, "purchase_intention", {}).values())
            if pi_vals:
                pis.append(sum(pi_vals) / len(pi_vals))
        avg_pi = (sum(pis) / len(pis)) if pis else 0.0
        avg_sentiment = (sum(sentiments) / len(sentiments)) if sentiments else 0.0
    return {"avg_purchase_intent": avg_pi, "avg_sentiment": avg_sentiment}


def calculate_network_metrics(network) -> Dict[str, Any]:
    """最小网络指标：密度与平均度。"""
    try:
        n = network.number_of_nodes()
        deg = dict(network.degree()) if n else {}
        avg_degree = (sum(deg.values()) / n) if n else 0.0
        return {"density": float(nx.density(network)), "avg_degree": float(avg_degree)}
    except Exception:
        return {"density": 0.0, "avg_degree": 0.0}


class MetricsNode(BaseNode):
    """t.3 Metrics calculation for the current time step.

    Computes brand/consumer/network metrics, plus core sandbox reputation metrics.
    """

    name = "MetricsNode"

    def __init__(self, model):
        self.model = model

    # ====== 微观层面：Agent内部状态向量（聚合） ======
    def _calculate_micro_states(self) -> Dict[str, Any]:
        # Get focal brand ID
        brands = self.model.agents_by_type.get("brand", [])
        bid = getattr(self.model, "focal_brand_id", None) or (brands[0].unique_id if brands else None)
        
        try:
            snapshot: List[Dict[str, Any]] = self.model._snapshot_consumer_states(brand_id=bid)
        except Exception as e:
            print(f"[MetricsNode] Error in _snapshot_consumer_states: {e}")
            import traceback
            traceback.print_exc()
            snapshot = []

        if not snapshot:
            print(f"[MetricsNode] Warning: Empty consumer snapshot for brand_id={bid}. Consumers count={len(self.model.agents_by_type.get('consumer', []))}")
            return {
                "competenceBelief_mean": 0.0,
                "valueCongruence_mean": 0.0,
                "emotionalConnection_mean": 0.0,
                "purchaseIntent_mean": 0.0,
                "advocacyIntent_mean": 0.0,
                "stance_distribution": {"supporter": 0.33, "neutral": 0.34, "opponent": 0.33},
            }

        cb, vc, ec, pi, ai = [], [], [], [], []
        stances: Dict[str, int] = {"supporter": 0, "neutral": 0, "opponent": 0}
        for s in snapshot:
            vec = s.get("dynamic_state_vector", {})
            cb.append(float(vec.get("competenceBelief", 0.0)))
            vc.append(float(vec.get("valueCongruence", 0.0)))
            ec.append(float(vec.get("emotionalConnection", 0.0)))
            pi.append(float(vec.get("purchaseIntent", 0.0)))
            ai.append(float(vec.get("advocacyIntent", 0.0)))
            # Lowercase stance for consistent counting
            # Use 'current_stance_global' if 'current_stance' is not present or if we want global stance
            # But snapshot usually provides 'current_stance' which is stance towards the brand.
            # If snapshot logic uses global, then it's fine. Let's check _snapshot_consumer_states in model.py later.
            
            # Try standard key first, then global fallback
            raw_stance = vec.get("current_stance") or vec.get("current_stance_global") or "neutral"
            stance = str(raw_stance).lower()
            
            # Handle potential mapping issues (e.g. if stance is 'support' instead of 'supporter')
            if "support" in stance: stance = "supporter"
            elif "oppon" in stance or "oppos" in stance: stance = "opponent"
            elif "neut" in stance: stance = "neutral"
            else: stance = "neutral"
            
            if stance in stances:
                stances[stance] += 1
            else:
                stances["neutral"] += 1

        n = max(1, len(snapshot))
        stance_dist = {k: (v / n) for k, v in stances.items()}
        return {
            "competenceBelief_mean": sum(cb) / n,
            "valueCongruence_mean": sum(vc) / n,
            "emotionalConnection_mean": sum(ec) / n,
            "purchaseIntent_mean": sum(pi) / n,
            "advocacyIntent_mean": sum(ai) / n,
            "stance_distribution": stance_dist,
        }

    # ====== 中观层面：沙盘状态指标（BRI/Polarization/SoV） ======
    def _calculate_market_metrics(self) -> Dict[str, Any]:
        # Get focal brand ID
        brands = self.model.agents_by_type.get("brand", [])
        bid = getattr(self.model, "focal_brand_id", None) or (brands[0].unique_id if brands else None)

        # BRI 当前步
        try:
            bri, nps = self.model._compute_bri_nps(brand_id=bid)
            print(f"[MetricsNode] Computed BRI: {bri}, NPS: {nps} for brand_id={bid}")
        except Exception as e:
            print(f"[MetricsNode] Error computing BRI/NPS: {e}")
            import traceback
            traceback.print_exc()
            bri = 0.0
            nps = 0.0

        # SoV 当前步（基于本步消息）
        try:
            # Use step buffer if available (captures all messages in step including distributed ones)
            # Otherwise fall back to current buffer (might be empty if distributed)
            msgs = list(getattr(self.model, "_last_step_messages", []) or [])
            if not msgs:
                msgs = list(getattr(self.model, "_message_buffer_step", []) or [])
            if not msgs:
                msgs = list(getattr(self.model, "_message_buffer_current", []) or [])
            sovr = self.model._compute_sov_from_messages(msgs)
            print(f"[MetricsNode] Computed SoV: {sovr}")
        except Exception as e:
            print(f"[MetricsNode] Error computing SoV: {e}")
            import traceback
            traceback.print_exc()
            sovr = {}

        # Esteban-Ray Polarization Index (discrete)
        # P = sum(pi_i * pi_j * |y_i - y_j|) * K
        # y: supporter=1.0, neutral=0.0, opponent=-1.0
        micro = self._calculate_micro_states()
        dist = micro.get("stance_distribution", {"supporter": 0.0, "neutral": 1.0, "opponent": 0.0})
        
        pi_s = max(0.0, float(dist.get("supporter", 0.0)))
        pi_n = max(0.0, float(dist.get("neutral", 0.0)))
        pi_o = max(0.0, float(dist.get("opponent", 0.0)))
        
        total = pi_s + pi_n + pi_o
        if total > 0:
            pi_s /= total
            pi_n /= total
            pi_o /= total
        else:
            pi_n = 1.0
        
        # Pairs: (S, N, |1-0|=1), (S, O, |1-(-1)|=2), (N, O, |0-(-1)|=1)
        # Symmetric sum in ER formula usually includes both (i,j) and (j,i).
        # Standard formula: P = K * sum_{i=1}^N sum_{j=1}^N pi_i * pi_j * |y_i - y_j|
        # Here N=3 groups.
        # Terms:
        # S-S: 0
        # S-N: pi_s * pi_n * 1
        # S-O: pi_s * pi_o * 2
        # N-S: pi_n * pi_s * 1
        # N-N: 0
        # N-O: pi_n * pi_o * 1
        # O-S: pi_o * pi_s * 2
        # O-N: pi_o * pi_n * 1
        # O-O: 0
        # Sum = 2 * (pi_s*pi_n + 2*pi_s*pi_o + pi_n*pi_o)
        
        raw_sum = 2.0 * (pi_s * pi_n * 1.0 + pi_s * pi_o * 2.0 + pi_n * pi_o * 1.0)
        
        # Normalization K
        # Max polarization occurs when 50% S, 50% O (bipolar).
        # pi_s=0.5, pi_o=0.5, pi_n=0.
        # Sum = 2 * (0 + 2*0.25 + 0) = 1.0.
        # So K=1.0 scales it to [0,1] naturally for this specific 3-point metric [-1, 0, 1].
        polarization_index = raw_sum
        polarization_index = max(0.0, min(1.0, polarization_index))

        return {
            "BRI": float(bri),
            "NPS": float(nps),
            "PolarizationIndex": 100.0 * float(polarization_index),
            "Brand_Current_SoV": 100.0 * float(sovr.get("Brand_Current_SoV", 0.0) or 0.0),
            "Competitor_Current_SoV": 100.0 * float(sovr.get("Competitor_Current_SoV", 0.0) or 0.0),
        }

    def run(self, state: SandboxState, **kwargs) -> SandboxState:
        metrics: Dict[str, Any] = {}
        try:
            metrics.update({"brand": calculate_brand_metrics(self.model)})
        except Exception:
            metrics["brand"] = {}
        try:
            metrics.update({"consumer": calculate_consumer_metrics(self.model)})
        except Exception:
            metrics["consumer"] = {}
        try:
            metrics.update({"network": calculate_network_metrics(self.model.network)})
        except Exception:
            metrics["network"] = {}

        # 沙盘核心指标（BRI/Polarization/SoV）与微观聚合
        try:
            metrics.update({"micro": self._calculate_micro_states()})
        except Exception:
            metrics["micro"] = {}
        try:
            metrics.update({"market": self._calculate_market_metrics()})
        except Exception:
            metrics["market"] = {}

        state.metrics = metrics
        state.logs.append(f"[MetricsNode] t={state.step} computed metrics keys={list(metrics.keys())}")
        return state
