"""Configuration loaders for dynamic sandbox runs.

Provides helpers to read scenario, population (JSONL), brand agent config,
calibration parameters, and simulation hyperparameters. Includes mapping from
Chinese driver labels to internal driver keys used by Consumer and Model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path


# Mapping: Chinese labels -> internal driver keys
DRIVER_CN_TO_KEY = {
    "价值导向": "value_oriented",
    "性能导向": "performance_oriented",
    "体验导向": "experience_oriented",
    "社会认同导向": "social_identity_oriented",
    "实用/安全导向": "practical_safety_oriented",
}

STANCE_CN_TO_KEY = {
    "支持者": "supporter",
    "中立": "neutral",
    "反对者": "opponent",
}

DRIVER_KEY_TO_CN = {v: k for k, v in DRIVER_CN_TO_KEY.items()}
STANCE_KEY_TO_CN = {v: k for k, v in STANCE_CN_TO_KEY.items()}


def _normalize_bri_floor(value: Any) -> float:
    """Return BRI threshold in the simulator's percentage-point scale."""
    value = float(value)
    if not __import__("math").isfinite(value):
        raise ValueError("Goal_BRI_Floor must be finite")
    return value * 100.0 if abs(value) <= 1.0 else value


def _normalize_nps_floor(value: Any) -> float:
    """Return NPS threshold in the simulator's [-1, 1] proportion scale."""
    value = float(value)
    if not __import__("math").isfinite(value):
        raise ValueError("Goal_NPS_Floor must be finite")
    return value / 100.0 if abs(value) > 1.0 else value

def load_json(path: str) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_scenario_records(path: str) -> List[Dict[str, Any]]:
    """Load one or more scenario records from JSON or JSONL.

    The released static subset uses ``description`` while the simulator's
    historical schema used ``scenario_description``.  Both are accepted and
    normalized here so a JSONL record cannot silently fall back to
    ``brand_awareness``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Scenario file does not exist: {p}")
    if p.suffix.lower() == ".jsonl":
        raw_items: List[Any] = []
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid scenario JSONL at {p}:{line_no}: {exc}") from exc
    else:
        raw = load_json(str(p))
        raw_items = raw if isinstance(raw, list) else [raw]
    records: List[Dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items, 1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Scenario record {index} in {p} must be a JSON object")
        record = dict(raw_item)
        description = (
            record.get("scenario_description")
            or record.get("description")
            or record.get("exam_text")
            or "brand_awareness"
        )
        record["scenario_description"] = str(description)
        record.setdefault("description", str(description))
        record.setdefault("scenario_id", f"scenario_{index:04d}")
        records.append(record)
    if not records:
        raise ValueError(f"Scenario file contains no records: {p}")
    return records


def load_scenario(path: Optional[str], index: int = 0) -> Dict[str, Any]:
    """Load one scenario JSON/JSONL record into pipeline background schema.

    Expected keys:
      - scenario_id: str
      - scenario_description or description: str
      - simulation_timesteps: int
    """
    if not path:
        return {
            "dominant_topic": "brand_awareness",
            "exam_text": "Benchmark",
            "market_signals": {},
            "constraints": {},
            "experiment_goals": {},
            "simulation_timesteps": 3,
        }
    records = load_scenario_records(path)
    if index < 0 or index >= len(records):
        raise IndexError(f"Scenario index {index} out of range for {path} ({len(records)} records)")
    raw = records[index]
    bg: Dict[str, Any] = {
        "dominant_topic": str(raw.get("scenario_description", "brand_awareness")),
        "exam_text": str(raw.get("scenario_description", "Benchmark")),
        "market_signals": {},
        "constraints": {},
        "experiment_goals": {
            "scenario_id": str(raw.get("scenario_id", "")),
            "scenario_index": index,
            "scenario_pool_size": len(records),
        },
    }
    # Attach timesteps hint for CLI override
    if isinstance(raw.get("simulation_timesteps"), int):
        bg["simulation_timesteps"] = int(raw["simulation_timesteps"])
    return bg


def load_population_jsonl(path: Optional[str]) -> Dict[str, Any]:
    """Load population spec from JSONL; returns dict with consumers and competitors.

    Each line should be a JSON with keys like:
      {"agent_id": "consumer_001", "agent_type": "consumer", "personality": {"driver": "性能导向", "initial_stance": "支持者"}, "initial_influence_score": 1}
      {"agent_id": "competitor_A", "agent_type": "competitor", "personality": {"strategy_archetype": "防御者 (The Defender)"}}
    """
    if not path:
        # Provide a minimal default population covering drivers/personas
        consumers: List[Dict[str, Any]] = []
        drivers = list(DRIVER_CN_TO_KEY.items())
        for i, (cn, key) in enumerate(drivers):
            consumers.append({
                "agent_id": f"consumer_{i+1:03d}",
                "driver_key": key,
                "driver_cn": cn,
                "initial_stance_cn": "中立",
                "initial_influence_score": 1.0,
            })
        competitors = [{"agent_id": "competitor_A", "persona": "defender"}]
        return {"consumers": consumers, "competitors": competitors}
    consumers: List[Dict[str, Any]] = []
    competitors: List[Dict[str, Any]] = []
    p = Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        atype = str(obj.get("agent_type", "")).lower()
        if atype == "consumer":
            pers = obj.get("personality", {}) or {}
            driver_cn = pers.get("driver")
            stance_cn = pers.get("initial_stance")
            consumers.append({
                "agent_id": obj.get("agent_id"),
                "driver_key": DRIVER_CN_TO_KEY.get(str(driver_cn), None),
                "driver_cn": driver_cn,
                "initial_stance_cn": stance_cn,
                "initial_influence_score": float(obj.get("initial_influence_score", 0.0)),
            })
        elif atype == "competitor":
            pers = obj.get("personality", {}) or {}
            arch = str(pers.get("strategy_archetype", "")).lower()
            # Map to internal persona keys
            if "defender" in arch:
                persona = "defender"
            elif "reactor" in arch:
                persona = "reactor"
            elif "analyzer" in arch:
                persona = "analyzer"
            elif "prospector" in arch:
                persona = "prospector"
            else:
                persona = "defender"
            competitors.append({
                "agent_id": obj.get("agent_id"),
                "persona": persona,
            })
    return {"consumers": consumers, "competitors": competitors}


def load_persona_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Persona config file does not exist: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Persona config must be a JSON object: {p}")
    return raw


def generate_population_from_persona_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    total = int(cfg.get("total_consumers", 100) or 100)
    mode = str(cfg.get("mode", "random")).lower()
    ratio = cfg.get("ratio") or []
    base_seed = int(cfg.get("base_seed", 7) or 7)
    seed_group_size = max(1, int(cfg.get("seed_group_size", 5) or 5))
    personas_file = cfg.get("personas_file")
    persona_seed_values = cfg.get("persona_seed_values")
    consumers: List[Dict[str, Any]] = []
    import random as _r

    rng = _r.Random(base_seed)

    def _normalize_driver_key(label: Any) -> str:
        text_label = str(label or "").strip()
        for cn, key in DRIVER_CN_TO_KEY.items():
            if cn and cn in text_label:
                return key
        if text_label in DRIVER_CN_TO_KEY.values():
            return text_label
        return next(iter(DRIVER_CN_TO_KEY.values()))

    def _normalize_stance_cn(label: Any) -> str:
        text_label = str(label or "").strip()
        for cn in STANCE_CN_TO_KEY.keys():
            if cn and cn in text_label:
                return cn
        for key, cn in STANCE_KEY_TO_CN.items():
            if key and key in text_label:
                return cn
        if "支" in text_label:
            return STANCE_KEY_TO_CN.get("supporter", next(iter(STANCE_CN_TO_KEY.keys())))
        if "反" in text_label:
            return STANCE_KEY_TO_CN.get("opponent", next(iter(STANCE_CN_TO_KEY.keys())))
        return STANCE_KEY_TO_CN.get("neutral", next(iter(STANCE_CN_TO_KEY.keys())))

    templates: List[Tuple[str, str]] = []
    if personas_file:
        persona_cfg = load_persona_config(personas_file)
        for arch in persona_cfg.get("archetypes", []) or []:
            driver_key = _normalize_driver_key(arch.get("driver") or arch.get("driver_cn"))
            stance_cn = _normalize_stance_cn(arch.get("stance") or arch.get("stance_cn"))
            templates.append((driver_key, stance_cn))
    if not templates:
        for d_key in DRIVER_CN_TO_KEY.values():
            for stance_cn in STANCE_CN_TO_KEY.keys():
                templates.append((d_key, stance_cn))

    def _seed_for_index(index: int) -> int:
        group_idx = index // seed_group_size
        cleaned: List[int] = []
        if isinstance(persona_seed_values, list):
            for val in persona_seed_values:
                try:
                    cleaned.append(int(val))
                except Exception:
                    continue
        if cleaned:
            return cleaned[group_idx % len(cleaned)]
        return base_seed + group_idx

    def _persona_label(driver_key: str, stance_cn: str) -> str:
        driver_cn = DRIVER_KEY_TO_CN.get(driver_key, driver_key)
        return f"{driver_cn}-{stance_cn}"

    def _build_consumer(agent_idx: int, driver_key: str, stance_cn: str) -> Dict[str, Any]:
        zero_based = agent_idx - 1
        return {
            "agent_id": f"consumer_{agent_idx:03d}",
            "driver_key": driver_key,
            "driver_cn": DRIVER_KEY_TO_CN.get(driver_key, driver_key),
            "initial_stance_cn": stance_cn,
            "initial_stance_key": STANCE_CN_TO_KEY.get(stance_cn, "neutral"),
            "initial_influence_score": 1.0 + 0.1 * rng.random(),
            "persona_seed": _seed_for_index(zero_based),
            "persona_seed_offset": zero_based % seed_group_size,
            "persona_label": _persona_label(driver_key, stance_cn),
        }

    if mode == "ratio" and isinstance(ratio, list) and ratio:
        spec_pairs: List[Tuple[str, str, int]] = []
        for item in ratio:
            driver_key = _normalize_driver_key(item.get("driver_cn") or item.get("driver"))
            stance_cn = _normalize_stance_cn(item.get("stance_cn") or item.get("stance"))
            pct = float(item.get("ratio", 0.0) or item.get("percent", 0.0))
            cnt = int(item.get("count", 0))
            if cnt <= 0:
                cnt = max(0, int(round(pct * total)))
            spec_pairs.append((driver_key, stance_cn, cnt))
        assigned = sum(x[2] for x in spec_pairs)
        remaining = max(0, total - assigned)
        while remaining > 0:
            if spec_pairs:
                driver_key, stance_cn, _ = rng.choice(spec_pairs)
            else:
                driver_key, stance_cn = rng.choice(templates)
            spec_pairs.append((driver_key, stance_cn, 1))
            remaining -= 1
        agent_idx = 1
        for driver_key, stance_cn, cnt in spec_pairs:
            for _ in range(max(0, cnt)):
                consumers.append(_build_consumer(agent_idx, driver_key, stance_cn))
                agent_idx += 1
        consumers = consumers[:total]
    else:
        pool: List[Tuple[str, str]] = []
        while len(pool) < total:
            shuffled = templates[:]
            rng.shuffle(shuffled)
            pool.extend(shuffled)
        agent_idx = 1
        for driver_key, stance_cn in pool[:total]:
            consumers.append(_build_consumer(agent_idx, driver_key, stance_cn))
            agent_idx += 1

    competitors = list(cfg.get("competitors", []) or [{"agent_id": "competitor_A", "persona": "defender"}])
    return {"consumers": consumers, "competitors": competitors}

def load_brand_config(path: Optional[str]) -> Dict[str, Any]:
    """Load brand agent config.

    Expected keys:
      - agent_id, agent_type == "brand"
      - model_path
      - identity_prompt_template (ignored by current implementation)
      - decision_trigger_config: {Goal_BRI_Floor}
    """
    if not path:
        return {
            "agent_id": "BrandAgent",
            "model_path": "gemini",
            "identity_prompt_template": "",
            "goal_bri_floor": 60.0,
            "goal_nps_floor": 0.2,
            "goal_rnsr_floor": 0.0,
        }
    raw = load_json(path)
    cfg = {
        "agent_id": raw.get("agent_id", "BrandAgent"),
        "model_path": raw.get("model_path", ""),
        "identity_prompt_template": raw.get("identity_prompt_template", ""),
        "goal_bri_floor": _normalize_bri_floor(raw.get("decision_trigger_config", {}).get("Goal_BRI_Floor", 60.0)),
        "goal_nps_floor": _normalize_nps_floor(raw.get("decision_trigger_config", {}).get("Goal_NPS_Floor", 0.2)),
        "goal_rnsr_floor": float(raw.get("decision_trigger_config", {}).get("Goal_RNSR_Floor", raw.get("goal_rnsr_floor", 0.0))),
    }
    return cfg


# Default driver sensitivities (ELM beta) and TPB intent weights.
# Rows are non-negative and sum to one: a positive message impact never
# lowers a belief dimension, and drivers differ in emphasis.
DEFAULT_BETA: Dict[str, List[List[float]]] = {
    "performance_oriented": [[0.9, 0.1, 0.0], [0.2, 0.6, 0.2], [0.0, 0.2, 0.8]],
    "value_oriented": [[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]],
    "experience_oriented": [[0.2, 0.2, 0.6], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7]],
    "social_identity_oriented": [[0.3, 0.3, 0.4], [0.2, 0.4, 0.4], [0.1, 0.3, 0.6]],
    "practical_safety_oriented": [[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.1, 0.2, 0.7]],
}
DEFAULT_W: Dict[str, Dict[str, List[float]]] = {
    "performance_oriented": {"purchase": [0.68, 0.11, 0.21], "advocacy": [0.30, 0.20, 0.50]},
    "value_oriented": {"purchase": [0.45, 0.35, 0.20], "advocacy": [0.25, 0.40, 0.35]},
    "experience_oriented": {"purchase": [0.20, 0.30, 0.50], "advocacy": [0.15, 0.35, 0.50]},
    "social_identity_oriented": {"purchase": [0.25, 0.35, 0.40], "advocacy": [0.20, 0.35, 0.45]},
    "practical_safety_oriented": {"purchase": [0.55, 0.25, 0.20], "advocacy": [0.40, 0.30, 0.30]},
}


def default_calibration() -> Dict[str, Any]:
    return {
        "beta": {key: [list(row) for row in rows] for key, rows in DEFAULT_BETA.items()},
        "w": {key: {kind: list(vec) for kind, vec in kinds.items()} for key, kinds in DEFAULT_W.items()},
        "source": "release_defaults",
    }


def _finite_matrix(rows: Any, shape: Tuple[int, int], label: str) -> List[List[float]]:
    import math
    try:
        matrix = [[float(x) for x in row] for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a {shape[0]}x{shape[1]} numeric matrix") from exc
    if len(matrix) != shape[0] or any(len(row) != shape[1] for row in matrix):
        raise ValueError(f"{label} must be a {shape[0]}x{shape[1]} numeric matrix")
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise ValueError(f"{label} contains NaN or Inf")
    return matrix


def load_calibration(path: Optional[str]) -> Dict[str, Any]:
    """Load ELM beta matrices and TPB weight vectors.

    Without a path the defaults above are used.  A calibration file follows
    this schema::

        {"elm_beta_coefficients": {"价值导向": {"competence": [..3..], "value": [..3..], "connection": [..3..]}, ...},
         "tpb_w_coefficients": {"价值导向": {"purchase_intent": [..3..], "advocacy_intent": [..3..]}, ...}}

    Drivers missing from the file keep the defaults.
    """
    calibration = default_calibration()
    if not path:
        return calibration
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {path}")
    for cn, mat in (raw.get("elm_beta_coefficients", {}) or {}).items():
        key = DRIVER_CN_TO_KEY.get(cn, cn if cn in DRIVER_CN_TO_KEY.values() else None)
        if not key or not isinstance(mat, dict):
            continue
        rows = [mat.get("competence"), mat.get("value"), mat.get("connection")]
        if any(row is None for row in rows):
            raise ValueError(f"elm_beta_coefficients[{cn}] needs competence/value/connection rows")
        calibration["beta"][key] = _finite_matrix(rows, (3, 3), f"elm_beta_coefficients[{cn}]")
    for cn, vecs in (raw.get("tpb_w_coefficients", {}) or {}).items():
        key = DRIVER_CN_TO_KEY.get(cn, cn if cn in DRIVER_CN_TO_KEY.values() else None)
        if not key or not isinstance(vecs, dict):
            continue
        for kind, field in (("purchase", "purchase_intent"), ("advocacy", "advocacy_intent")):
            if field in vecs:
                calibration["w"][key][kind] = _finite_matrix([vecs[field]], (1, 3), f"tpb_w_coefficients[{cn}].{field}")[0]
    calibration["source"] = str(path)
    return calibration


def normalize_hyperparams(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize a hyperparameter mapping while preserving extension keys.

    Defaults follow Appendix C / Table 11 of the paper (bold column):
    base_p 0.10, virality 0.10, topic_coeff 0.10, feed_cap 20, noise_delta 0.10,
    decay_lambda 1.00, comp_pressure 1.00.  ``consumer_update_rate`` (the base
    λ_u), ``memory_volatility`` (0 = the paper's noise-free Eq. 1), the stance
    thresholds, and the stance-alignment factors of Eq. (10) are release
    operationalizations documented in the README.
    """
    raw = dict(raw or {})
    decay = float(raw.get("memory_decay_global_lambda", raw.get("memory_decay_lambda", 1.0)))
    heat = float(raw.get("platform_heat_decay_gamma", raw.get("platform_heat_decay_factor", 0.9)))
    thresholds = raw.get("stance_derivation_thresholds", {}) or {}
    hp = {
        **raw,
        "memory_decay_global_lambda": decay,
        "memory_decay_lambda": decay,
        "consumer_update_rate": float(raw.get("consumer_update_rate", 0.15)),
        "memory_volatility": float(raw.get("memory_volatility", 0.0)),
        "platform_base_p": float(raw.get("platform_base_p", raw.get("base_push_p", 0.1))),
        "platform_virality_c1": float(raw.get("platform_virality_c1", 0.1)),
        "platform_topic_c2": float(raw.get("platform_topic_c2", 0.1)),
        "platform_heat_decay_factor": heat,
        "platform_heat_decay_gamma": heat,
        "feed_capacity": int(raw.get("feed_capacity", 20)),
        "initial_influence_noise_delta": float(raw.get("initial_influence_noise_delta", 0.10)),
        "competitor_pressure_mu": float(raw.get("competitor_pressure_mu", 1.0)),
        "stance_alignment_neutral": float(raw.get("stance_alignment_neutral", 0.75)),
        "stance_alignment_opposed": float(raw.get("stance_alignment_opposed", 0.5)),
        "group_influence_gamma": {},
        # Polarization stance groups coincide with the NPS promoter/detractor
        # groups: advocacy > 0.7 or < 0.3 on the paper's [0, 1] scale equals
        # > 0.4 or < -0.4 on the internal tanh scale.
        "stance_derivation_thresholds": {
            "promoter_threshold": float(thresholds.get("promoter_threshold", 0.4)),
            "detractor_threshold": float(thresholds.get("detractor_threshold", -0.4)),
        },
    }
    ggamma = raw.get("group_influence_gamma", {}) or {}
    for cn, val in ggamma.items():
        key = DRIVER_CN_TO_KEY.get(cn) or (cn if cn in DRIVER_CN_TO_KEY.values() else None)
        if not key:
            continue
        try:
            hp["group_influence_gamma"][key] = float(val)
        except Exception:
            continue
    if not hp["group_influence_gamma"]:
        hp["group_influence_gamma"] = {key: 1.0 for key in DRIVER_CN_TO_KEY.values()}
    return hp


def load_hyperparams(path: Optional[str]) -> Dict[str, Any]:
    """Load simulation hyperparameters with explicit path validation."""
    if not path:
        return normalize_hyperparams({})
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Hyperparameter file does not exist: {p}")
    raw = load_json(str(p))
    if not isinstance(raw, dict):
        raise ValueError(f"Hyperparameter config must be a JSON object: {p}")
    return normalize_hyperparams(raw)


def generate_alignment_report(
    population_spec: Dict[str, Any],
    brand_cfg: Dict[str, Any],
    calibration: Dict[str, Any],
    hyperparams: Dict[str, Any],
    spec_pdf_text: Optional[str] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"ok": True}

    consumers = list(population_spec.get("consumers", []) or [])
    competitors = list(population_spec.get("competitors", []) or [])

    invalid_consumers: List[str] = []
    missing_consumer_fields: List[str] = []
    for c in consumers:
        if c.get("driver_key") is None and c.get("driver_cn"):
            invalid_consumers.append(str(c.get("agent_id")))
        for k in ("agent_id", "initial_stance_cn", "initial_influence_score"):
            if k not in c:
                missing_consumer_fields.append(k)

    missing_competitor_fields: List[str] = []
    for comp in competitors:
        for k in ("agent_id", "persona"):
            if k not in comp:
                missing_competitor_fields.append(k)

    brand_required = ("agent_id", "model_path", "goal_bri_floor", "goal_rnsr_floor")
    missing_brand: List[str] = [k for k in brand_required if k not in brand_cfg]

    beta = calibration.get("beta", {}) or {}
    w = calibration.get("w", {}) or {}
    drivers = list(DRIVER_CN_TO_KEY.values())
    missing_beta: List[str] = [d for d in drivers if d not in beta]
    missing_w: List[str] = [d for d in drivers if d not in w]

    hp_required = (
        "memory_decay_lambda",
        "platform_virality_c1",
        "platform_topic_c2",
        "platform_heat_decay_factor",
        "stance_derivation_thresholds",
    )
    missing_hp: List[str] = [k for k in hp_required if k not in hyperparams]

    report["population"] = {
        "invalid_consumers": invalid_consumers,
        "missing_consumer_fields": sorted(set(missing_consumer_fields)),
        "missing_competitor_fields": sorted(set(missing_competitor_fields)),
    }
    report["brand_config"] = {"missing_fields": missing_brand}
    report["calibration"] = {"missing_beta": missing_beta, "missing_w": missing_w}
    report["hyperparams"] = {"missing_fields": missing_hp}

    pdf_terms = None
    ok_pdf_terms = True
    if spec_pdf_text:
        terms: Dict[str, List[str]] = {
            "drivers_cn": list(DRIVER_CN_TO_KEY.keys()),
            "ELM": ["ELM", "精细加工可能性模型"],
            "TPB": ["TPB", "计划行为理论"],
            "BRI": ["BRI", "品牌声誉指数"],
            "SoV": ["SoV", "声音份额"],
            "RNSR": ["RNSR", "归一化净情感比"],
            "Polarization": ["极化", "Polarization"],
            "MemoryDecay": ["记忆衰减", "记忆衰退", "遗忘", "memory"],
            "PlatformVirality": [
                "平台传播",
                "平台扩散",
                "传播性",
                "扩散",
                "病毒性",
                "virality",
                "传播参数",
                "扩散系数",
            ],
            "StanceThresholds": ["阈值", "门槛", "threshold"],
        }
        present: Dict[str, bool] = {}
        missing_labels: List[str] = []
        text = str(spec_pdf_text)
        for label, variants in terms.items():
            found = False
            for v in variants:
                if v and v in text:
                    found = True
                    break
            present[label] = found
            if not found:
                missing_labels.append(label)
        pdf_terms = {"present": present, "missing": missing_labels}
        ok_pdf_terms = len(missing_labels) == 0

    report["pdf_terms"] = pdf_terms
    report["ok_pdf_terms"] = ok_pdf_terms

    ok = not (
        invalid_consumers
        or missing_consumer_fields
        or missing_competitor_fields
        or missing_brand
        or missing_beta
        or missing_w
        or missing_hp
    ) and ok_pdf_terms
    report["ok"] = bool(ok)
    return report


def extract_pdf_text(path: str, max_chars: int = 200000) -> Optional[str]:
    try:
        import fitz  # type: ignore
        doc = fitz.open(path)
        parts: List[str] = []
        for pg in doc:
            try:
                parts.append(pg.get_text())
            except Exception:
                continue
        text = "\n".join(parts)
        return text[:max_chars]
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        text = extract_text(path)
        if text:
            return text[:max_chars]
    except Exception:
        pass
    try:
        import PyPDF2  # type: ignore
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts: List[str] = []
            for page in getattr(reader, "pages", []) or []:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n".join(parts)
            if text:
                return text[:max_chars]
    except Exception:
        pass
    return None
def load_unified_config(path: Optional[str]) -> Dict[str, Any]:
    """读取统一配置文件。

    统一配置示例：
    {
      "background_texts": ["...一句话..."],
      "steps": 10,
      "num_consumers": 50,
      "num_competitors": 1,
      "llm_model": "gemini-2.5-flash-lite",
      "consumer_persona_code": "R",  # R=随机15人格分布
      "competitor_persona_code": "D"  # D=defender, A=analyzer, P=prospector, R=reactor
    }
    若路径为空或解析失败，返回空字典。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Unified config file does not exist: {p}")
    raw = load_json(str(p))
    if isinstance(raw, list):
        return {"background_texts": [str(x) for x in raw]}
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unified config must be a JSON object or list: {p}")

def build_population_from_unified_config(
    ucfg: Dict[str, Any],
    consumers_persona_path: Optional[str] = None,
    competitors_persona_path: Optional[str] = None,
) -> Dict[str, Any]:
    """根据统一配置生成人口规范。

    - 消费者：支持 R=随机（覆盖15人格随机分布）；默认按随机生成总数。
    - 竞争者：按代码映射生成指定数量与人格。
    """
    total = int(ucfg.get("num_consumers", 100) or 100)
    c_code = str(ucfg.get("consumer_persona_code", "R") or "R").upper()
    # 使用既有生成器（ratio/random），此处采用随机模式
    base_seed = int(ucfg.get("seed", 7) or 7)
    consumers_spec = []
    try:
        base_dir = Path(__file__).resolve().parents[1]
        personas_path = consumers_persona_path or str(base_dir / "input" / "personas_consumers.json")
        consumers_spec = generate_population_from_persona_config({
            "total_consumers": total,
            "mode": "random",
            "base_seed": base_seed,
            "personas_file": personas_path,
            "seed_group_size": 5,
        }).get("consumers", [])
    except Exception:
        consumers_spec = []

    # 竞争者映射
    comp_n = int(ucfg.get("num_competitors", 1) or 1)
    comp_code_raw = str(ucfg.get("competitor_persona_code", "D") or "D").upper()
    code_map = {"D": "defender", "A": "analyzer", "P": "prospector", "R": "reactor"}
    
    competitors_spec = []
    if "," in comp_code_raw:
        codes = [c.strip() for c in comp_code_raw.split(",") if c.strip()]
        # 限制最多2个，如用户所述
        if len(codes) > 2:
            codes = codes[:2]
        for i, code in enumerate(codes):
             persona = code_map.get(code, "defender")
             competitors_spec.append({"agent_id": f"competitor_{i+1:02d}", "persona": persona})
    else:
        persona = code_map.get(comp_code_raw, "defender")
        for i in range(max(1, comp_n)):
            competitors_spec.append({"agent_id": f"competitor_{i+1:02d}", "persona": persona})

    return {"consumers": consumers_spec, "competitors": competitors_spec}
