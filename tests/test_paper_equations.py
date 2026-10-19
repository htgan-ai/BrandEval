"""Checks that tie the simulator to the paper's equations and protocol.

Eq. (1)/(11): V_t = (1 - lambda_u) V_{t-1} + lambda_u * gamma_t * tanh(beta_driver x_t)
Eq. (9)/(10): P_push and P_ret = P_push * s_u * a_{m,u}
Section 4.3: the 5 x 3 persona cells start inside their stance band.
Appendix C.1.2 / D.4: evidence-consistency gate and Backfire Rate.
"""
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from brandeval.agents.brand_agent import BrandAgent
from brandeval.agents.brand_agent_ablation import BrandAgentNoSR
from brandeval.agents.consumer import CORE_DRIVERS
from brandeval.cli.run_sandbox import parse_args
from brandeval.config.loaders import generate_population_from_persona_config, load_calibration, load_hyperparams
from brandeval.model import BrandEvalModel
from brandeval.pipeline.nodes.simulation_node import SimulationNode

STATE_KEYS = ("capability_belief", "value_resonance", "emotional_bond")


def _state(consumer, brand_id):
    return np.array([consumer.cbbe_state[brand_id][key] for key in STATE_KEYS], dtype=float)


def _one_consumer_per_driver(seed=3, stance="neutral"):
    stance_cn = {"neutral": "中立", "supporter": "支持者", "opponent": "反对者"}[stance]
    spec = {
        "consumers": [
            {"agent_id": f"c{i}", "driver_key": driver, "driver_cn": "", "initial_stance_cn": stance_cn,
             "initial_stance_key": stance, "initial_influence_score": 1.0}
            for i, driver in enumerate(CORE_DRIVERS)
        ],
        "competitors": [],
    }
    model = BrandEvalModel(num_consumers=len(CORE_DRIVERS), num_brands=1, seed=seed, population_spec=spec,
                           calibration=load_calibration(None), hyperparams=load_hyperparams(None))
    focal = list(model.agents_by_type["brand"])[0].unique_id
    return model, list(model.agents_by_type["consumer"]), focal


def test_cascade_update_is_equation_11():
    _, consumers, focal = _one_consumer_per_driver()
    x = np.array([0.4, 0.2, -0.1])
    gamma = 0.5
    for consumer in consumers:
        consumer.memory_volatility = 0.0
        consumer.cbbe_state[focal] = dict(zip(STATE_KEYS, (0.1, -0.2, 0.3)))
        beta = np.array(consumer._get_beta_for_driver(consumer.core_driver), dtype=float)
        lam = consumer._update_rate()
        expected = (1 - lam) * np.array([0.1, -0.2, 0.3]) + lam * gamma * np.tanh(beta @ x)
        consumer._cascade_update(focal, tuple(x), verification_factor=gamma)
        assert np.allclose(_state(consumer, focal), expected, atol=1e-12), consumer.core_driver
        w_a = np.array(consumer._get_w_for_driver(consumer.core_driver, "advocacy"), dtype=float)
        assert consumer.intent_state[focal]["advocacy_intent"] == pytest.approx(math.tanh(float(w_a @ expected)))


def test_default_update_rate_is_table_11_product():
    _, consumers, _ = _one_consumer_per_driver()
    hp = load_hyperparams(None)
    assert hp["memory_decay_global_lambda"] == 1.0
    assert hp["memory_volatility"] == 0.0
    assert set(hp["group_influence_gamma"].values()) == {1.0}
    assert hp["stance_derivation_thresholds"] == {"promoter_threshold": 0.4, "detractor_threshold": -0.4}
    for consumer in consumers:
        assert consumer.learning_rate_lambda == pytest.approx(0.15)
        assert consumer.memory_volatility == 0.0
        assert consumer._update_rate() == pytest.approx(0.15)


def test_verified_positive_post_raises_every_driver_and_unverified_never_flips_sign():
    _, consumers, focal = _one_consumer_per_driver()
    impact = {"capability": 0.3, "value": 0.2, "emotion": 0.2}
    for consumer in consumers:
        consumer.memory_volatility = 0.0
        before = _state(consumer, focal)
        consumer.receive_message({"message_type": "official_post", "sender_role": "brand",
                                  "content": {"content": "x", "brand_id": focal, "info_impact": impact, "verification_factor": 1.0}})
        consumer._process_messages()
        after = _state(consumer, focal)
        assert np.all(after > before), consumer.core_driver
        consumer.receive_message({"message_type": "official_post", "sender_role": "brand",
                                  "content": {"content": "y", "brand_id": focal, "info_impact": impact, "verification_factor": 0.0}})
        consumer._process_messages()
        decayed = _state(consumer, focal)
        # gamma = 0: the message contributes nothing; memory only decays toward zero.
        assert np.all(np.abs(decayed) <= np.abs(after) + 1e-12)
        assert np.all(np.sign(decayed) * np.sign(after) >= 0)


def test_initial_persona_stances_are_realized(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    spec = generate_population_from_persona_config({"total_consumers": 30, "base_seed": 7})
    spec["competitors"] = []
    model = BrandEvalModel(num_consumers=30, num_brands=1, seed=7, population_spec=spec,
                           calibration=load_calibration(None), hyperparams=load_hyperparams(None))
    focal = list(model.agents_by_type["brand"])[0].unique_id
    consumers = list(model.agents_by_type["consumer"])
    assert sorted(c.current_stance_global for c in consumers).count("supporter") == 10
    assert all(c.stance_by_brand[focal] == c.current_stance_global for c in consumers)
    model.start_benchmark("synthetic crisis text", brand_id=focal)
    assert all(c.stance_by_brand[focal] == c.current_stance_global for c in consumers)
    # 10 promoters and 10 detractors at t = 0: NPS = 0 points, not a collapsed neutral population.
    metrics = model.jsonl_records[-1]["market_metrics"]
    assert metrics["NPS"] == pytest.approx(0.0)
    assert metrics["PolarizationIndex"] > 0.0


def test_exposure_follows_equation_10():
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=7, hyperparams=load_hyperparams(None))
    focal = list(model.agents_by_type["brand"])[0].unique_id
    model.focal_brand_id = focal
    consumer = list(model.agents_by_type["consumer"])[0]
    brand_post = {"sender_id": focal, "sender_role": "brand", "message_type": "official_post",
                  "content": {"content": "x", "brand_id": focal}}
    consumer.stance_by_brand[focal] = "supporter"
    assert model._stance_alignment(brand_post, consumer) == 1.0
    consumer.stance_by_brand[focal] = "neutral"
    assert model._stance_alignment(brand_post, consumer) == 0.75
    consumer.stance_by_brand[focal] = "opponent"
    assert model._stance_alignment(brand_post, consumer) == 0.5
    ugc = {"sender_id": "other", "sender_role": "consumer", "message_type": "ugc_post",
           "content": {"content": "y", "brand_id": focal, "authorStance": "opponent"}}
    assert model._stance_alignment(ugc, consumer) == 1.0
    consumer.stance_by_brand[focal] = "supporter"
    assert model._stance_alignment(ugc, consumer) == 0.5
    # P_ret = P_push * s_u * a: a broadcast reaches a fully susceptible aligned consumer and never an s_u = 0 one.
    consumer.influence_susceptibility = 1.0
    consumer.message_queue = []
    model._message_buffer_prev = [brand_post]
    model.distribute_information()
    assert len(consumer.message_queue) == 1
    consumer.influence_susceptibility = 0.0
    consumer.message_queue = []
    model._message_buffer_prev = [brand_post]
    model.distribute_information()
    assert consumer.message_queue == []


def test_push_probability_is_equation_9():
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=7, hyperparams=load_hyperparams(None))
    expected = 0.1 * (1 + 0.1 * math.log(3.0 + 1) + 0.1 * math.log(5.0 + 1))
    assert model._compute_effective_push_p(influence=3.0, heat=5.0) == pytest.approx(expected)


def test_default_calibration_is_non_negative_and_row_normalized():
    calibration = load_calibration(None)
    assert calibration["source"] == "release_defaults"
    for driver in CORE_DRIVERS:
        beta = np.array(calibration["beta"][driver], dtype=float)
        assert beta.shape == (3, 3) and np.all(beta >= 0)
        assert np.allclose(beta.sum(axis=1), 1.0)
        for kind in ("purchase", "advocacy"):
            w = np.array(calibration["w"][driver][kind], dtype=float)
            assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)


def test_evidence_gate_is_fractional_membership():
    node = object.__new__(SimulationNode)
    node.evidence_store = {"report-1": {}, "report-2": {}}
    assert node._verify_action_claims([]) == 1.0
    assert node._verify_action_claims(["report-1", "report-2"]) == 1.0
    assert node._verify_action_claims(["report-1", "report-x"]) == 0.5
    assert node._verify_action_claims(["nope"]) == 0.0


def test_lexicon_rater_is_deterministic_and_bounded():
    from brandeval.pipeline.message_impact import MessageImpactRater

    rater = MessageImpactRater()
    text = "我们已公开第三方检测报告，启动免费召回并向用户致歉。"
    first, second = rater.rate(text, "brand"), rater.rate(text, "brand")
    assert first == second
    assert all(-1.0 <= v <= 1.0 for v in first.values())
    assert first["capability"] > 0 and first["value"] > 0 and first["emotion"] > 0
    hostile = rater.rate("这些都是谣言，我们将追究造谣者的法律责任。", "brand")
    assert hostile["emotion"] < 0 and hostile["capability"] <= 0
    assert rater.rate("", "brand") == {"capability": 0.0, "value": 0.0, "emotion": 0.0}


def test_message_impact_is_independent_of_the_agents_self_report(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    from brandeval.pipeline.nodes.base_node import SandboxState

    background = {"exam_text": "synthetic event", "dominant_topic": "safety"}
    impacts = []
    for claimed in (-1.0, 1.0):
        node = SimulationNode(num_consumers=1, num_brands=1, seed=11)
        node.enable_agent_llms(model_name="offline", brand_model_name="offline",
                               competitor_model_name="offline", consumer_model_name="offline")
        action = {"final_action": {
            "final_action_content": "我们已公开第三方检测报告并启动免费召回。",
            "claimed_evidence": [],
            "predicted_impact": {"capability": claimed, "value": claimed, "emotion": claimed},
        }}
        node._brand_agent.strategic_decision = lambda brief, action=action: action
        node.run(SandboxState(step=1, background_info=background))
        posts = [m for m in node.model.public_log_last_round() if m.get("sender_role") == "brand"]
        assert len(posts) == 1
        impacts.append(posts[0]["content"]["info_impact"])
    assert impacts[0] == impacts[1]
    assert impacts[0]["capability"] > 0  # rated from the text, not from the self-report


def test_brand_observation_exposes_only_public_text():
    import json

    node = object.__new__(SimulationNode)
    hidden = {"sender_role": "consumer", "message_type": "ugc_post", "content": {
        "content": "这个品牌太差了，失望。", "brand_id": "b",
        "info_impact": {"capability": 0.931415}, "llm_prompt": "SECRET_PROMPT", "verification_factor": 1.0}}
    node.model = SimpleNamespace(_last_step_messages=[hidden], _compute_sov_from_messages=lambda _: {},
                                 _dominant_topic_snapshot=lambda _: {})
    brief = node._build_brand_brief(1)
    dumped = json.dumps(brief, ensure_ascii=False)
    assert "0.931415" not in dumped and "SECRET_PROMPT" not in dumped and "info_impact" not in dumped
    assert brief["key_negative_posts"] == ["这个品牌太差了，失望。"]
    agent = BrandAgent(SimpleNamespace(_last_step_messages=[hidden]), goal_bri_floor=60.0, goal_nps_floor=0.2)
    prompt = agent._build_decision_prompt(1, brief["dominant_topic"], brief["key_negative_posts"], brief["metrics"])
    assert "0.931415" not in prompt and "SECRET_PROMPT" not in prompt


def test_competitor_move_from_previous_round_is_observed(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    from brandeval.pipeline.nodes.base_node import SandboxState

    node = SimulationNode(num_consumers=1, num_brands=1, seed=11, hyperparams={"competitor_pressure_mu": 1.0},
                          population_spec={"consumers": [], "competitors": [{"agent_id": "competitor_01", "persona": "defender"}]})
    node.enable_agent_llms(model_name="offline", brand_model_name="offline",
                           competitor_model_name="offline", consumer_model_name="offline")
    node._competitors[0].strategic_decision = lambda brief: {"final_action": {"final_action_content": "UNIQUE_COMPETITOR_MOVE"}}
    background = {"exam_text": "synthetic event", "dominant_topic": "safety"}
    node.run(SandboxState(step=1, background_info=background))
    brief = node._build_brand_brief(2)
    assert brief["competitor_last_move"] == "UNIQUE_COMPETITOR_MOVE"
    assert node._brand_agent._get_competitor_last_move() == "UNIQUE_COMPETITOR_MOVE"
    roles = {str(m.get("sender_role")) for m in node.model.public_log_last_round()}
    assert {"brand", "competitor", "consumer"} <= roles


def test_backfire_rate_counts_attenuated_official_rounds(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=7)
    model.start_benchmark("synthetic event")
    assert model.jsonl_records[-1]["market_metrics"]["BackfireRate"] is None
    focal = model.focal_brand_id
    model._posts_current = [{"authorRole": "brand", "verification_factor": 0.5}]
    model._write_jsonl_record(t=1, brand_id=focal, source_messages=[])
    assert model.jsonl_records[-1]["market_metrics"]["BackfireRate"] == pytest.approx(1.0)
    model._posts_current = [{"authorRole": "brand", "verification_factor": 1.0}]
    model._write_jsonl_record(t=2, brand_id=focal, source_messages=[])
    assert model.jsonl_records[-1]["market_metrics"]["BackfireRate"] == pytest.approx(0.5)
    assert "ConsistencyRate" not in model.jsonl_records[-1]["market_metrics"]


def test_no_sr_ablation_shares_the_publication_path_but_not_the_sr_schema():
    assert issubclass(BrandAgentNoSR, BrandAgent)
    sr = set(BrandAgent._decision_json_schema(None)["properties"])
    no_sr = set(BrandAgentNoSR._decision_json_schema(None)["properties"])
    workflow = {"goal", "stakeholder_summary", "candidate_plans", "selected_plan"}
    assert workflow <= sr
    assert not workflow & no_sr
    assert {"final_action_content", "claimed_evidence", "predicted_impact"} <= no_sr


def test_cli_brand_policy_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["brandeval-sandbox", "--brand-policy", "no_sr"])
    assert parse_args().brand_policy == "no_sr"
    monkeypatch.setattr(sys, "argv", ["brandeval-sandbox"])
    assert parse_args().brand_policy == "sr"


def test_brand_alerts_are_diagnostic_only():
    agent = BrandAgent(SimpleNamespace(), goal_bri_floor=60.0, goal_nps_floor=0.2)
    assert agent._get_bri_nps({}) == (None, None)
    alerts = agent._check_alerts(None, None, "", [])
    assert alerts["trigger"] is False and alerts["diagnostic_only"] is True


def test_rater_network_flag_follows_repository_semantics(monkeypatch):
    from brandeval.pipeline.message_impact import MessageImpactRater

    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    assert MessageImpactRater.from_model_name("gemini-2.5-flash-lite").llm is None
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "0")
    assert MessageImpactRater.from_model_name("gemini-2.5-flash-lite").llm is not None
    assert MessageImpactRater.from_model_name("offline").llm is None
    monkeypatch.delenv("BRANDEVAL_API_DISABLE_NETWORK")
    monkeypatch.delenv("GEMINI_API_DISABLE_NETWORK", raising=False)
    assert MessageImpactRater.from_model_name("gemini-2.5-flash-lite").llm is None  # disabled unless explicitly "0"


def test_published_text_is_rated_after_truncation(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    from brandeval.pipeline.nodes.base_node import SandboxState
    from brandeval.pipeline.message_impact import MessageImpactRater

    visible = ("我们已公开第三方检测报告并启动免费召回。" * 400)[:5000]
    long_text = visible + "这些都是谣言，我们将追究造谣者的法律责任。" * 20   # suffix beyond the publication limit
    node = SimulationNode(num_consumers=1, num_brands=1, seed=11)
    node.enable_agent_llms(model_name="offline", brand_model_name="offline",
                           competitor_model_name="offline", consumer_model_name="offline")
    node._brand_agent.strategic_decision = lambda brief: {"final_action": {"final_action_content": long_text, "claimed_evidence": []}}
    node.run(SandboxState(step=1, background_info={"exam_text": "synthetic event", "dominant_topic": "safety"}))
    post = [m for m in node.model.public_log_last_round() if m.get("sender_role") == "brand"][0]
    assert len(post["content"]["content"]) == 5000
    assert post["content"]["info_impact"] == MessageImpactRater().rate(visible, "brand")

