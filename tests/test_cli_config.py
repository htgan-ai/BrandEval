"""CLI configuration precedence and explicit-input checks."""

import json
import sys
from pathlib import Path

import pytest

from brandeval.cli.run_sandbox import _resolve_cli_configuration, main, parse_args
from brandeval.config.loaders import load_calibration, load_hyperparams


def _args(monkeypatch, *items):
    monkeypatch.setattr(sys, "argv", ["brandeval-sandbox", *items])
    return parse_args()


def test_validate_only_does_not_create_output(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["brandeval-sandbox", "--validate-only", "--background", "{}"])
    main()
    out = capsys.readouterr().out
    payload = json.loads(out[out.find("{"):])
    assert payload["resolved_config"]["steps"] == 3
    assert not list(tmp_path.iterdir())


def test_scenario_background_and_cli_precedence(monkeypatch, tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({
        "scenario_id": "s1",
        "scenario_description": "scenario topic",
        "simulation_timesteps": 8,
    }), encoding="utf-8")
    background = tmp_path / "background.json"
    background.write_text(json.dumps({
        "exam_text": "background text",
        "dominant_topic": "background topic",
    }), encoding="utf-8")
    args = _args(monkeypatch, "--scenario-file", str(scenario), "--background-file", str(background), "--steps", "2")
    resolved = _resolve_cli_configuration(args)
    assert resolved["steps"] == 2
    assert resolved["background"]["exam_text"] == "background text"
    assert resolved["background"]["dominant_topic"] == "background topic"


def test_population_file_controls_default_quantity_and_cli_overrides(monkeypatch, tmp_path):
    population = tmp_path / "population.jsonl"
    rows = [
        {"agent_id": "c1", "agent_type": "consumer", "personality": {"driver": "价值导向", "initial_stance": "中立"}},
        {"agent_id": "c2", "agent_type": "consumer", "personality": {"driver": "性能导向", "initial_stance": "支持者"}},
    ]
    population.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    args = _args(monkeypatch, "--population-file", str(population))
    resolved = _resolve_cli_configuration(args)
    assert resolved["num_consumers"] == 2
    assert resolved["resolved"]["population_source"] == "population_file"

    args = _args(monkeypatch, "--population-file", str(population), "--num-consumers", "5")
    assert _resolve_cli_configuration(args)["num_consumers"] == 5


def test_persona_quantity_and_mutual_exclusion(monkeypatch, tmp_path):
    persona = tmp_path / "persona.json"
    persona.write_text(json.dumps({"total_consumers": 4, "base_seed": 7}), encoding="utf-8")
    args = _args(monkeypatch, "--persona-config-file", str(persona))
    assert _resolve_cli_configuration(args)["num_consumers"] == 4
    args = _args(monkeypatch, "--persona-config-file", str(persona), "--population-file", str(persona))
    with pytest.raises(ValueError, match="either --population-file"):
        _resolve_cli_configuration(args)


def test_explicit_paths_fail_instead_of_fallback(monkeypatch, tmp_path):
    args = _args(monkeypatch, "--unified-config", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        _resolve_cli_configuration(args)
    args = _args(monkeypatch, "--spec-text-file", str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        _resolve_cli_configuration(args)


def test_spec_text_and_temperature_rounds(monkeypatch, tmp_path):
    spec = tmp_path / "spec.txt"
    spec.write_text("ELM TPB BRI SoV RNSR Polarization memory virality threshold 价值导向", encoding="utf-8")
    args = _args(monkeypatch, "--spec-text-file", str(spec), "--dialogue-rounds", "3", "--llm-temp-min", "0.2", "--llm-temp-max", "0.4")
    resolved = _resolve_cli_configuration(args)
    assert resolved["dialogue_rounds"] == 3
    assert resolved["temp_min"] == 0.2
    assert resolved["temp_max"] == 0.4
    assert resolved["alignment"]["pdf_terms"] is not None


def test_custom_calibration_is_authoritative(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "elm_beta_coefficients": {"价值导向": {"competence": [9, 0, 0], "value": [0, 8, 0], "connection": [0, 0, 7]}},
        "tpb_w_coefficients": {"价值导向": {"purchase_intent": [1, 2, 3], "advocacy_intent": [4, 5, 6]}},
    }, ensure_ascii=False), encoding="utf-8")
    loaded = load_calibration(str(calibration))
    assert loaded["beta"]["value_oriented"][0] == [9, 0, 0]
    assert loaded["w"]["value_oriented"]["purchase"] == [1, 2, 3]


def test_hyperparam_defaults_and_override(tmp_path):
    defaults = load_hyperparams(None)
    assert defaults["competitor_pressure_mu"] == 1.0
    assert defaults["platform_heat_decay_gamma"] == 0.9
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"competitor_pressure_mu": 1.5, "retention_probability": 0.7}), encoding="utf-8")
    loaded = load_hyperparams(str(hp))
    assert loaded["competitor_pressure_mu"] == 1.5
    assert loaded["retention_probability"] == 0.7
