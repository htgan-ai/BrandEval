"""CLI entry to run the BrandEval sandbox pipeline (package-local).

Usage:
  python -m brandeval.cli.run_sandbox --background "{\"dominant_topic\": \"new_product\"}" --steps 3
  python -m brandeval.cli.run_sandbox --background-file path/to/bg.json --steps 5
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import random
from pathlib import Path
from typing import Any, Dict

from brandeval.pipeline.nodes.strategy_node import StrategyNode
from brandeval.pipeline.nodes.simulation_node import SimulationNode
from brandeval.pipeline.nodes.metrics_node import MetricsNode
from brandeval.pipeline.nodes.summary_node import SummaryNode
from brandeval.pipeline.nodes.dialogue_node import DialogueNode
from brandeval.pipeline.orchestrator import Orchestrator
from brandeval.evaluation.runner import EvaluationRunner
from brandeval.evaluation.visualization import generate_metrics_plots
from brandeval.llm.interface import GeminiChatLLM
from brandeval.config.loaders import generate_alignment_report
from brandeval.config.loaders import extract_pdf_text
from brandeval.config.loaders import (
    load_scenario,
    load_population_jsonl,
    load_unified_config,
    build_population_from_unified_config,
    load_persona_config,
    generate_population_from_persona_config,
    load_brand_config,
    load_calibration,
    load_hyperparams,
    normalize_hyperparams,
)
from brandeval.config.loaders import DRIVER_CN_TO_KEY


def _set_global_seeds(seed: int) -> None:
    """Deterministically seed all RNG sources used by the sandbox."""
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            try:
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run BrandEval sandbox pipeline")
    parser.add_argument("--background", type=str, default="", help="JSON string background info")
    parser.add_argument("--background-file", type=str, default="", help="Path to JSON file for background info")
    parser.add_argument("--unified-config", type=str, default="", help="Path to unified sandbox config JSON")
    parser.add_argument("--scenario-file", type=str, default="", help="Path to scenario JSON or JSONL config")
    parser.add_argument("--scenario-index", type=int, default=0, help="Zero-based record index when --scenario-file is JSONL")
    parser.add_argument("--population-file", type=str, default="", help="Path to population JSONL config")
    parser.add_argument("--persona-config-file", type=str, default="", help="Path to persona config JSON for 15 archetypes")
    parser.add_argument("--brand-config-file", type=str, default="", help="Path to brand agent JSON config")
    parser.add_argument("--calibration-file", type=str, default="", help="Path to calibration JSON config")
    parser.add_argument("--hyperparams-file", type=str, default="", help="Path to simulation hyperparams JSON config")
    parser.add_argument("--hyperparams", type=str, default="", help="Path to hyperparams JSON file or JSON string")
    parser.add_argument("--steps", type=int, default=None, help="Number of time steps to run")
    parser.add_argument("--spec-pdf", type=str, default="", help="Path to spec PDF to align")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configs against spec")
    parser.add_argument("--spec-text-file", type=str, default="", help="Path to plain text extracted from spec PDF")
    parser.add_argument("--verbose", action="store_true", help="Print per-step posts and metrics")
    parser.add_argument("--num-consumers", type=int, default=None, help="Number of consumer agents to simulate")
    parser.add_argument("--dialogue-rounds", type=int, default=None, help="Rounds for DialogueNode at t.0 stage")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for simulation")
    parser.add_argument("--llm-temp-min", type=float, default=0.35, help="Min temperature for consumer LLM")
    parser.add_argument("--llm-temp-max", type=float, default=0.85, help="Max temperature for consumer LLM")
    parser.add_argument("--slim-output", action="store_true", help="Only output per-round messages and metrics")
    parser.add_argument("--brand-policy", choices=["sr", "no_sr"], default="sr", help="Focal brand policy: Strategic Rationale workflow (sr) or the single-action ablation (no_sr)")
    parser.add_argument("--output", type=str, help="Explicit output directory name")
    return parser.parse_args()


def _translate_benchmark_background(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Translate user-provided Benchmark JSON into pipeline background schema.

    Expected Benchmark JSON example:
    {
      "task1_input": { "scenario_description": "...", "opinion_texts": ["...", "..."] },
      "task2_input": { "scenario_description": "...", "reputation_goal": "..." },
      "task3_input": { "scenario_description": "...", "action_taken": "...", "prediction_goal": "..." }
    }

    Output background keys used by the pipeline:
      - dominant_topic: short string (uses scenario_description)
      - exam_text: initial broadcast content at t=0 (scenario + opinions)
      - market_signals: optional extras (empty by default)
      - constraints: optional goals (e.g., reputation_goal)
      - experiment_goals: optional metadata (action_taken, prediction_goal)
    """
    try:
        has_benchmark = any(k in raw for k in ("task1_input", "task2_input", "task3_input"))
        if not has_benchmark:
            return raw

        t1 = isinstance(raw.get("task1_input"), dict) and raw.get("task1_input") or {}
        t2 = isinstance(raw.get("task2_input"), dict) and raw.get("task2_input") or {}
        t3 = isinstance(raw.get("task3_input"), dict) and raw.get("task3_input") or {}

        scenario = str(
            t1.get("scenario_description")
            or t2.get("scenario_description")
            or t3.get("scenario_description")
            or ""
        )
        opinions = t1.get("opinion_texts") if isinstance(t1.get("opinion_texts"), list) else []

        exam_lines = []
        if scenario:
            exam_lines.append(scenario)
        for o in opinions:
            try:
                exam_lines.append(str(o))
            except Exception:
                continue
        exam_text = "\n".join(exam_lines).strip() or "Benchmark"

        bg: Dict[str, Any] = {
            "dominant_topic": scenario or "brand_awareness",
            "exam_text": exam_text,
            "market_signals": {},
            "constraints": {},
            "experiment_goals": {},
        }

        if t2.get("reputation_goal"):
            bg["constraints"]["reputation_goal"] = str(t2["reputation_goal"])
        if t3.get("action_taken"):
            bg["experiment_goals"]["action_taken"] = str(t3["action_taken"])
        if t3.get("prediction_goal"):
            bg["experiment_goals"]["prediction_goal"] = str(t3["prediction_goal"])

        return bg
    except Exception:
        # On any parsing error, fall back to raw to avoid blocking
        return raw


def _require_file(path: str, label: str) -> Path:
    """Resolve an explicit input path and fail instead of silently falling back."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{label} does not exist: {p}")
    return p


def _load_json_object(path: str, label: str) -> Dict[str, Any]:
    p = _require_file(path, label)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to parse {label}: {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object: {p}")
    return raw


def _parse_hyperparams_arg(value: str) -> Dict[str, Any]:
    """Parse an inline JSON object or an explicit JSON file path."""
    text = str(value or "").strip()
    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid inline --hyperparams JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Inline --hyperparams must be a JSON object")
        return normalize_hyperparams(raw)
    return load_hyperparams(str(_require_file(text, "Hyperparameter file")))


def _resolve_cli_configuration(args) -> Dict[str, Any]:
    """Resolve all CLI inputs before constructing agents or output directories.

    Precedence is explicit CLI scalar override, then scenario/background input,
    then unified config/defaults.  Population source precedence is
    ``--population-file`` over ``--persona-config-file`` over generated defaults;
    an explicit ``--num-consumers`` controls the requested target size.
    """
    if args.population_file and args.persona_config_file:
        raise ValueError("Use either --population-file or --persona-config-file, not both")
    if args.spec_pdf and args.spec_text_file:
        raise ValueError("Use either --spec-pdf or --spec-text-file, not both")

    unified = load_unified_config(args.unified_config) if args.unified_config else {}

    scenario_background: Dict[str, Any] = {}
    if args.scenario_file:
        _require_file(args.scenario_file, "Scenario file")
        scenario_background = load_scenario(args.scenario_file, index=int(args.scenario_index))

    file_background: Dict[str, Any] = {}
    if args.background_file:
        raw_background = _load_json_object(args.background_file, "Background file")
        file_background = _translate_benchmark_background(raw_background)
        # The bundled sandbox file is a unified config with background_texts;
        # accept that shape as well as a direct background object.
        if isinstance(raw_background.get("background_texts"), list):
            file_background["exam_text"] = "\n".join(str(x) for x in raw_background["background_texts"])
        for key in (
            "steps", "num_consumers", "num_competitors", "seed", "brand_name",
            "product_name", "brand_llm_model", "consumer_llm_model",
            "competitor_llm_model", "dialogue_rounds", "goal_bri_floor",
            "goal_nps_floor", "goal_rnsr_floor", "consumer_persona_code",
            "competitor_persona_code",
        ):
            if key in raw_background and key not in unified:
                unified[key] = raw_background[key]

    background: Dict[str, Any] = {
        "dominant_topic": "",
        "exam_text": "\n".join(str(x) for x in unified.get("background_texts", []))
        if isinstance(unified.get("background_texts"), list)
        else str(unified.get("exam_text", "Benchmark")),
        "market_signals": {},
        "constraints": {},
        "experiment_goals": {},
    }
    # Unified config may also carry background metadata; keep it below the
    # dedicated scenario/background files in precedence.
    for key in (
        "dominant_topic", "exam_text", "market_signals", "constraints",
        "experiment_goals", "brand_id", "brand_keywords", "competitor_keywords",
    ):
        if key in unified:
            background[key] = unified[key]
    # Scenario is the lowest-priority explicit background source; a direct
    # background file and then --background override it.
    background.update(scenario_background)
    background.update(file_background)
    if args.background:
        try:
            cli_background = json.loads(args.background)
        except Exception as exc:
            raise ValueError(f"Invalid --background JSON: {exc}") from exc
        if not isinstance(cli_background, dict):
            raise ValueError("--background must be a JSON object")
        background.update(_translate_benchmark_background(cli_background))

    # Explicit CLI steps win over scenario, unified config, and default.
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        steps = int(args.steps)
    elif isinstance(background.get("simulation_timesteps"), int):
        steps = int(background["simulation_timesteps"])
    else:
        steps = int(unified.get("steps", 3))
    if steps < 1:
        raise ValueError("Resolved steps must be positive")
    unified["steps"] = steps
    background["simulation_timesteps"] = steps

    # Population source and quantity precedence.
    source = "generated"
    if args.population_file:
        population_spec = load_population_jsonl(str(_require_file(args.population_file, "Population file")))
        source = "population_file"
        source_count = len(population_spec.get("consumers", []) or [])
    elif args.persona_config_file:
        persona_cfg = load_persona_config(str(_require_file(args.persona_config_file, "Persona config file")))
        source = "persona_config"
        if args.num_consumers is not None:
            if args.num_consumers < 1:
                raise ValueError("--num-consumers must be positive")
            persona_cfg["total_consumers"] = int(args.num_consumers)
        elif "total_consumers" not in persona_cfg:
            persona_cfg["total_consumers"] = int(unified.get("num_consumers", 100))
        population_spec = generate_population_from_persona_config(persona_cfg)
        source_count = len(population_spec.get("consumers", []) or [])
    else:
        if args.num_consumers is not None:
            if args.num_consumers < 1:
                raise ValueError("--num-consumers must be positive")
            unified["num_consumers"] = int(args.num_consumers)
        population_spec = build_population_from_unified_config(unified)
        source_count = len(population_spec.get("consumers", []) or [])

    if args.num_consumers is not None:
        if args.num_consumers < 1:
            raise ValueError("--num-consumers must be positive")
        num_consumers = int(args.num_consumers)
    elif source in {"population_file", "persona_config"}:
        num_consumers = source_count
    else:
        num_consumers = int(unified.get("num_consumers", 100))
    if num_consumers < 1:
        raise ValueError("Resolved num_consumers must be positive")
    unified["num_consumers"] = num_consumers

    brand_cfg = load_brand_config(args.brand_config_file or None)
    if unified.get("brand_name"):
        brand_cfg["brand_name"] = unified["brand_name"]
    if unified.get("product_name"):
        brand_cfg["product_name"] = unified["product_name"]
    for key in ("goal_bri_floor", "goal_nps_floor", "goal_rnsr_floor"):
        if isinstance(unified.get(key), (int, float)):
            brand_cfg[key] = float(unified[key])
    calibration = load_calibration(args.calibration_file or None)
    hyperparams = load_hyperparams(None)
    if args.hyperparams:
        hyperparams.update(_parse_hyperparams_arg(args.hyperparams))
    elif args.hyperparams_file:
        hyperparams.update(load_hyperparams(args.hyperparams_file))
    hyperparams = normalize_hyperparams(hyperparams)

    pdf_text: Optional[str] = None
    if args.spec_text_file:
        pdf_text = _require_file(args.spec_text_file, "Spec text file").read_text(encoding="utf-8")
    elif args.spec_pdf:
        spec_path = _require_file(args.spec_pdf, "Spec PDF")
        pdf_text = extract_pdf_text(str(spec_path))
        if not pdf_text:
            raise ValueError(f"Unable to extract text from spec PDF: {spec_path}")

    alignment = generate_alignment_report(
        population_spec=population_spec or {},
        brand_cfg=brand_cfg or {},
        calibration=calibration or {},
        hyperparams=hyperparams or {},
        spec_pdf_text=pdf_text,
    )
    background["spec_alignment"] = alignment
    dialogue_rounds = args.dialogue_rounds if args.dialogue_rounds is not None else unified.get("dialogue_rounds", 1)
    dialogue_rounds = int(dialogue_rounds)
    if dialogue_rounds < 1:
        raise ValueError("Resolved dialogue_rounds must be positive")
    temp_min = float(args.llm_temp_min)
    temp_max = float(args.llm_temp_max)
    if temp_min < 0 or temp_max < 0 or temp_min > temp_max:
        raise ValueError("LLM temperature bounds must satisfy 0 <= min <= max")

    resolved = {
        "steps": steps,
        "num_consumers": num_consumers,
        "seed": int(args.seed if args.seed is not None else unified.get("seed", 7)),
        "dialogue_rounds": dialogue_rounds,
        "llm_temp_min": temp_min,
        "llm_temp_max": temp_max,
        "brand_policy": str(getattr(args, "brand_policy", "sr") or "sr"),
        "population_source": source,
        "population_count": source_count,
        "scenario_file": str(args.scenario_file or ""),
        "scenario_index": int(args.scenario_index),
        "background_file": str(args.background_file or ""),
        "unified_config": str(args.unified_config or ""),
        "spec_pdf": str(args.spec_pdf or ""),
        "spec_text_file": str(args.spec_text_file or ""),
        "population_file": str(args.population_file or ""),
        "persona_config_file": str(args.persona_config_file or ""),
        "brand_config_file": str(args.brand_config_file or ""),
        "calibration_file": str(args.calibration_file or ""),
        "hyperparams_file": str(args.hyperparams_file or ""),
        "background": background,
        "hyperparams": hyperparams,
        "alignment": alignment,
    }
    return {
        "unified": unified,
        "background": background,
        "population_spec": population_spec,
        "brand_cfg": brand_cfg,
        "calibration": calibration,
        "hyperparams": hyperparams,
        "alignment": alignment,
        "resolved": resolved,
        "num_consumers": num_consumers,
        "steps": steps,
        "dialogue_rounds": dialogue_rounds,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "seed": resolved["seed"],
    }


def load_background(args) -> Dict[str, Any]:
    # Scenario first
    scenario_bg = load_scenario(args.scenario_file, index=int(getattr(args, "scenario_index", 0))) if getattr(args, "scenario_file", "") else {}
    if args.background_file:
        path = Path(args.background_file)
        raw = json.loads(path.read_text(encoding="utf-8"))
        merged = _translate_benchmark_background(raw)
        if scenario_bg:
            merged = {**scenario_bg, **merged}
        return merged
    if args.background:
        s = args.background
        try:
            raw = json.loads(s)
        except Exception:
            try:
                if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
                    raw = json.loads(s[1:-1])
                else:
                    raw = json.loads(s)
            except Exception:
                raw = {}
        merged = _translate_benchmark_background(raw)
        if scenario_bg:
            merged = {**scenario_bg, **merged}
        return merged
    # Default minimal background
    bg = {
        "dominant_topic": "brand_awareness",
        "market_signals": {"trend": "neutral"},
        "constraints": {"budget_cap": 20000},
    }
    if scenario_bg:
        bg = {**scenario_bg, **bg}
    return bg


def main():
    args = parse_args()
    config = _resolve_cli_configuration(args)
    unified = config["unified"]
    background = config["background"]
    population_spec = config["population_spec"]
    brand_cfg = config["brand_cfg"]
    calibration = config["calibration"]
    hyperparams = config["hyperparams"]
    alignment = config["alignment"]
    steps = config["steps"]
    seed_value = config["seed"]
    _set_global_seeds(seed_value)
    dialogue_rounds = config["dialogue_rounds"]
    temp_min = config["temp_min"]
    temp_max = config["temp_max"]

    if args.validate_only:
        print(json.dumps({"resolved_config": config["resolved"], "alignment": alignment}, ensure_ascii=False, indent=2))
        if not alignment.get("ok", False):
            raise SystemExit(1)
        return

    consumer_llm_model = str(unified.get("consumer_llm_model", unified.get("llm_model", "gemini-2.5-flash-lite")))
    brand_llm_model = str(unified.get("brand_llm_model", consumer_llm_model))
    competitor_llm_model = str(unified.get("competitor_llm_model", consumer_llm_model))

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%m%d%H%M")
        output_dir = Path(f"./output{timestamp}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config["resolved"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "run.log"
    try:
        log_path.write_text("", encoding="utf-8")
    except Exception:
        pass

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('[%H:%M:%S]')} {message}"
        print(line)
        try:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            pass

    log("=== BrandEval Sandbox Run ===")
    log(f"Output directory: {output_dir.resolve()}")
    log(f"Consumer LLM: {consumer_llm_model} | Brand LLM: {brand_llm_model} | Competitor LLM: {competitor_llm_model}")
    log(f"Configured steps: {unified.get('steps', 3)} | Requested consumers: {unified.get('num_consumers', 200)} | Requested competitors: {unified.get('num_competitors', 1)}")

    gemini_adapter = GeminiChatLLM(model_name=brand_llm_model)
    
    # === Auto-extract dominant topic if missing ===
    current_topic = background.get("dominant_topic", "")
    
    if not current_topic or not str(current_topic).strip():
        log("Dominant topic missing or empty. Extracting from exam_text using LLM...")
        try:
            exam_text = background.get("exam_text", "")
            prompt_content = (
                f"请阅读以下背景文本，并提取3-4个用户讨论的焦点小主题（Sub-topics），这些主题应反映舆论的具体切入点（如‘安全隐患’、‘品牌态度’、‘赔偿方案’等），而非宏大的抽象概念。\n"
                f"同时，请提取5-8个辅助关键词。\n"
                f"文本：{exam_text[:2000]}\n"
                f"输出格式：主题1 | 主题2 | 主题3 | 关键词1, 关键词2, 关键词3"
            )
            messages = [{"role": "user", "content": prompt_content}]
            extracted = gemini_adapter.generate(messages, response_format="text")
            
            if extracted:
                # Clean up
                cleaned = extracted.replace("，", ",").replace("\n", "").strip()
                cleaned = cleaned.replace('"', '').replace("'", "")
                
                focus_topics = []
                keywords = []
                
                # Split by pipe for topics vs keywords
                # Expect: T1 | T2 | T3 | K1, K2...
                parts = [p.strip() for p in cleaned.split("|") if p.strip()]
                
                if len(parts) >= 2:
                    # Last part is likely keywords
                    kw_part = parts[-1]
                    # Check if it looks like keywords (contains commas)
                    if "," in kw_part:
                        keywords = [k.strip() for k in kw_part.split(",") if k.strip()]
                        focus_topics = parts[:-1]
                    else:
                        # Maybe all are topics? Or all keywords?
                        # Assume if multiple pipes, they are topics
                        focus_topics = parts
                else:
                    # Fallback: just keywords
                    keywords = [k.strip() for k in cleaned.split(",") if k.strip()]
                    if keywords:
                        focus_topics = [keywords[0]]

                # Join topics for display/storage
                final_topic_str = ", ".join(focus_topics)
                
                background["dominant_topic"] = final_topic_str
                background["topic_keywords"] = keywords
                log(f"Extracted focus topics: {final_topic_str}")
                log(f"Extracted keywords: {keywords}")
            else:
                log("LLM returned empty topic.")
                background["dominant_topic"] = ""
                background["topic_keywords"] = []
        except Exception as e:
            log(f"Topic extraction failed: {e}")
            background["dominant_topic"] = ""
            background["topic_keywords"] = []
    else:
        log(f"Using existing topic: {current_topic}")
    # ==============================================
    # Fallback: 若主题仍为空，用 exam_text 前30字填充，避免后续启发式失效
    if not str(background.get("dominant_topic", "")).strip():
        fallback_topic = str(background.get("exam_text", ""))[:30] or "brand_awareness"
        background["dominant_topic"] = fallback_topic
        log(f"Fallback dominant_topic set to: {fallback_topic}")

    strategy = StrategyNode(llm=gemini_adapter, agent_ids=["brand"])
    steps = int(unified.get("steps", 3))

    simulation = SimulationNode(
        num_consumers=int(unified.get("num_consumers", 200)),
        num_brands=1,
        network_type="small_world",
        seed=seed_value,
        population_spec=population_spec or None,
        calibration=calibration or None,
        hyperparams=hyperparams or None,
        brand_agent_cfg=brand_cfg or None,
        background_context=background.get("exam_text", ""),
        brand_policy=str(getattr(args, "brand_policy", "sr") or "sr"),
    )
    try:
        simulation.enable_agent_llms(
            model_name=consumer_llm_model,
            temp_min=temp_min,
            temp_max=temp_max,
            brand_model_name=brand_llm_model,
            competitor_model_name=competitor_llm_model,
            consumer_model_name=consumer_llm_model,
        )
    except Exception:
        pass
    try:
        model = simulation.model
        consumers = list(model.agents_by_type.get("consumer", []))
        competitors = list(model.agents_by_type.get("competitor", []))
        brands = list(model.agents_by_type.get("brand", []))
        log(f"Initialized agents: consumers={len(consumers)} brands={len(brands)} competitors={len(competitors)}")
        if consumers:
            driver_counts: Dict[str, int] = {}
            for cons in consumers:
                driver = getattr(cons, "core_driver", "unknown")
                driver_counts[driver] = driver_counts.get(driver, 0) + 1
            ratios = ", ".join(f"{k}:{v/len(consumers):.2%}" for k, v in driver_counts.items())
            log(f"Consumer driver distribution: {ratios}")
    except Exception:
        pass

    metrics = MetricsNode(model=simulation.model)
    summary = SummaryNode(llm=gemini_adapter)
    dialogue = DialogueNode(llm=gemini_adapter, rounds=dialogue_rounds)
    orch = Orchestrator(
        strategy=strategy,
        simulation=simulation,
        metrics=metrics,
        summary=summary,
        output_dir=str(output_dir),
        dialogue=dialogue,
    )
    runner = EvaluationRunner(orchestrator=orch, output_dir=str(output_dir), verbose=args.verbose, slim_output=args.slim_output, log_fn=log)

    log(f"Background dominant topic: {background.get('dominant_topic')}")
    sample_exam = str(background.get("exam_text", ""))[:200].replace("\n", " ")
    log(f"Background exam_text preview: {sample_exam}")

    try:
        result_dir = runner.run(background_info=background, steps=steps)
        try:
            generate_metrics_plots(result_dir / "metrics_record.json", result_dir)
        except Exception as e:
            log(f"Plotting failed: {e}")
        log(f"Run complete. Results in: {result_dir}")
    except Exception as e:
        import traceback
        log(f"Run failed with error: {e}")
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
