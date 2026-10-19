"""Offline contract checks for metrics, agents, data, and release integrity."""
import math
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from brandeval.model import BrandEvalModel
from brandeval.agents.consumer import Consumer
from brandeval.agents.brand_agent import BrandAgent
from brandeval.agents.base import BrandEvalAgent
from brandeval.config.loaders import generate_population_from_persona_config
from brandeval.config.loaders import load_scenario
from brandeval.evaluation.runner import EvaluationRunner
from brandeval.pipeline.nodes.simulation_node import SimulationNode
from brandeval.cli.run_sandbox import parse_args


def runner_metrics(values, nps=-25.0, sov=60.0, polarization=25.0):
    # Round records carry BRI/NPS in points and SoV/Polarization in percent.
    records = [{"timestep": i + 1, "market_metrics": {
        "BRI": value, "NPS": nps, "SoV": sov, "PolarizationIndex": polarization
    }} for i, value in enumerate(values)]
    runner = object.__new__(EvaluationRunner)
    runner.orchestrator = SimpleNamespace(model=SimpleNamespace(jsonl_records=records))
    return runner._compute_dynamic_metrics()


def test_summary_units_and_arv():
    out = runner_metrics([10, 0, 10])
    assert out['Final_BRI'] == 10
    assert out['Final_NPS'] == -25
    assert out['Final_SoV'] == 60
    assert out['Final_PolarizationIndex'] == 25
    assert out['ARV'] == pytest.approx(100 * math.sqrt(2 / 9))
    assert out['CRS'] == 1
    assert out['units']['Final_NPS'] == 'points' and out['units']['Final_SoV'] == 'percent'


@pytest.mark.parametrize('nps, sov, pol', [(1.0, 0.5, 1.0), (-1.0, 1.0, 0.5), (0.0, 0.0, 0.0)])
def test_small_final_values_are_not_rescaled(nps, sov, pol):
    # One promoter in 100 consumers is NPS = 1 point; one focal post in 200 is SoV = 0.5 %.
    out = runner_metrics([5, 5], nps=nps, sov=sov, polarization=pol)
    assert out['Final_NPS'] == nps
    assert out['Final_SoV'] == sov
    assert out['Final_PolarizationIndex'] == pol


def test_full_pipeline_units_reach_the_summary(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    model = BrandEvalModel(num_consumers=4, num_brands=1, seed=7)
    model.start_benchmark("synthetic event")
    model.step()
    last = model.jsonl_records[-1]["market_metrics"]
    runner = object.__new__(EvaluationRunner)
    runner.orchestrator = SimpleNamespace(model=model)
    out = runner._compute_dynamic_metrics()
    assert out['Final_NPS'] == last['NPS']
    assert out['Final_SoV'] == last['SoV']
    assert out['Final_PolarizationIndex'] == last['PolarizationIndex']


@pytest.mark.parametrize('series, expected', [([10, 0, 5], 2), ([0, 0, 0], 1), ([10, 5, 0], 1)])
def test_crs_strictly_after_bottom_and_censoring(series, expected):
    assert runner_metrics(series)['CRS'] == expected


def test_arv_affine_invariance_and_constant():
    assert runner_metrics([-12, -20, -8])['ARV'] == pytest.approx(runner_metrics([4, 0, 6])['ARV'])
    assert runner_metrics([3, 3, 3])['ARV'] == 0



def test_arv_crs_exclude_initialization_record():
    runner = object.__new__(EvaluationRunner)
    runner.orchestrator = SimpleNamespace(model=SimpleNamespace(jsonl_records=[
        {"timestep": 0, "market_metrics": {"BRI": 999.0, "NPS": 0.0}},
        {"timestep": 1, "market_metrics": {"BRI": 10.0, "NPS": 0.0}},
        {"timestep": 2, "market_metrics": {"BRI": 0.0, "NPS": 0.0}},
    ]))
    out = runner._compute_dynamic_metrics()
    assert out["trajectory_points"] == 2
    assert out["ARV"] == pytest.approx(50.0)


def test_uniform_bri_weights():
    model = BrandEvalModel(num_consumers=3, num_brands=1, seed=7)
    assert model.bri_weights == (1 / 3, 1 / 3, 1 / 3)
    bid = model.agents_by_type['brand'][0].unique_id
    for c in model.agents_by_type['consumer']:
        c.cbbe_state[bid] = dict(capability_belief=1., value_resonance=0., emotional_bond=-.4)
    bri, _ = model._compute_bri_nps(bid)
    assert bri == pytest.approx(20.)


def test_population_is_fifteen_cells():
    p = generate_population_from_persona_config({'total_consumers': 30, 'base_seed': 7})
    cells = [(c['driver_key'], c['initial_stance_key']) for c in p['consumers']]
    assert len(set(cells)) == 15
    assert all(cells.count(cell) == 2 for cell in set(cells))


def test_neutral_stance_alias_is_normalized():
    assert BrandEvalModel._normalize_stance_value(None, '中立者', None) == 'neutral'


def test_cli_omitted_overrides(monkeypatch):
    monkeypatch.setattr('sys.argv', ['brandeval-sandbox'])
    args = parse_args()
    assert args.num_consumers is None and args.seed is None and args.steps is None


def test_paper_nps_literal_thresholds():
    # Internal tanh state .2 maps to .6 on the paper's [0,1] scale.
    c = SimpleNamespace(cbbe_state={'b': dict(capability_belief=0., value_resonance=0., emotional_bond=0.)}, intent_state={'b': {'advocacy_intent': .2}})
    model = SimpleNamespace(agents_by_type={'consumer': [c]}, bri_weights=(1/3,)*3)
    _, nps = BrandEvalModel._compute_bri_nps(model, 'b')
    assert nps == 0  # .6 is neither >.7 nor <.3 under the paper's literal definition


def test_published_nps_uses_point_scale():
    model = BrandEvalModel(num_consumers=2, num_brands=1, seed=7)
    bid = model.agents_by_type['brand'][0].unique_id
    for c, advocacy in zip(model.agents_by_type['consumer'], (0.8, 0.2)):
        c.intent_state[bid]['advocacy_intent'] = advocacy
    _, nps = model._compute_bri_nps(bid)
    assert nps == pytest.approx(50.0)


def test_fabricated_report_is_not_verified():
    node = object.__new__(SimulationNode)
    node.evidence_store = {'verified-report-001': {}}
    assert node._verify_action_claims(['fabricated-report-999']) == 0


def test_observation_does_not_read_latent_metrics():
    def forbid(*args, **kwargs):
        raise AssertionError('latent metrics read')
    model = SimpleNamespace(_last_step_messages=[], _compute_sov_from_messages=lambda _: {},
                            _compute_bri_nps=forbid, _dominant_topic_snapshot=lambda _: {})
    node = object.__new__(SimulationNode)
    node.model = model
    brief = node._build_brand_brief(1)
    assert not {'BRI', 'NPS'} & set(brief['metrics'])


def test_sr_schema_fields():
    schema = BrandAgent._decision_json_schema(None)
    assert {'goal','stakeholder_summary','candidate_plans','selected_plan','claimed_evidence'} <= set(schema['properties'])


def test_official_message_shape():
    c = object.__new__(Consumer)
    c.message_queue = [{'message_type':'official_post', 'sender_role':'brand',
                        'content': {'content':'Evidence released', 'brand_id':'b', 'info_impact': {}}}]
    updates = []
    c._cascade_update = lambda *args, **kwargs: updates.append(args)
    c._process_messages()
    assert len(updates) == 1


def test_one_message_one_queue_entry():
    c = object.__new__(Consumer)
    c.message_queue = []
    c.receive_message({'message_type':'ugc_post','content':'x'})
    assert len(c.message_queue) == 1


def test_feed_capacity_is_enforced():
    model = BrandEvalModel(num_consumers=3, num_brands=1, seed=7, hyperparams={'feed_capacity': 2})
    model._message_buffer_prev = [{'sender_id':'brand-x','sender_role':'brand','message_type':'official_post','content':{'content':str(i),'brand_id':'b'}} for i in range(5)]
    model.distribute_information()
    assert all(len(c.message_queue) <= 2 for c in model.agents_by_type['consumer'])


def test_episode_ids_independent_of_previous_runs():
    a = BrandEvalAgent(SimpleNamespace(), 'probe')
    b = BrandEvalAgent(SimpleNamespace(), 'probe')
    assert a.unique_id == b.unique_id


def test_static_jsonl_scenario_is_normalized_for_single_run():
    root = Path(__file__).resolve().parents[1]
    scenario = load_scenario(str(root / "data" / "partial" / "static_scenario_descriptions_sanitized.jsonl"))
    assert scenario["experiment_goals"]["scenario_id"].startswith("static_")
    assert scenario["exam_text"]
    assert scenario["dominant_topic"]


def test_brand_observes_latest_public_competitor_move():
    from brandeval.agents.brand_agent import BrandAgent

    agent = object.__new__(BrandAgent)
    agent.model = SimpleNamespace(_last_step_messages=[
        {"sender_role": "competitor", "content": {"content": "竞品公开说明"}},
        {"sender_role": "consumer", "content": "消费者跟进"},
    ])
    assert agent._get_competitor_last_move() == "竞品公开说明"


def test_brand_trigger_uses_normalized_kpi_units():
    agent = BrandAgent(SimpleNamespace(), goal_bri_floor=0.6, goal_nps_floor=20.0)
    assert agent.goal_bri_floor == pytest.approx(60.0)
    assert agent.goal_nps_floor == pytest.approx(0.2)
    assert agent._check_alerts(80.0, 0.5, "", [])['trigger'] is False
    alerts = agent._check_alerts(40.0, 0.5, "", [])
    assert alerts['trigger'] is True
    # Alerts are diagnostics; the focal agent acts every round (Section 4.2).
    assert alerts['diagnostic_only'] is True


def test_strategy_rejects_nonfinite_and_unnormalized_budget_values():
    from brandeval.pipeline.nodes.strategy_node import StrategyNode

    node = StrategyNode()
    proposal = node._heuristic_strategy({"dominant_topic": "topic"})
    proposal["budget"]["total"] = float("nan")
    proposal["budget"]["allocation"] = {"social": 2.0}
    result = node._validate_strategy(proposal)
    assert result["valid"] is False
    assert "budget.total_not_number" in result["errors"]
    assert "budget.allocation_not_normalized" in result["errors"]
