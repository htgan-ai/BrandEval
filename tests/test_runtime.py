from __future__ import annotations

from brandeval.model import BrandEvalModel
from brandeval.pipeline.nodes.base_node import SandboxState
from brandeval.pipeline.nodes.simulation_node import SimulationNode
from types import SimpleNamespace


def test_mesa_step_counter_is_single_increment_and_focal_is_fixed():
    model = BrandEvalModel(num_consumers=2, num_brands=1, seed=7)
    model.start_benchmark("synthetic event")
    assert model.focal_brand_id == "brand_1"
    for expected in (1, 2, 3):
        model.step()
        assert model.steps == expected
        assert model.jsonl_records[-1]["timestep"] == expected


def test_competitor_role_cannot_be_selected_as_focal_brand():
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=7)
    model.agents_by_type["brand"].append(SimpleNamespace(unique_id="competitor_1", role="competitor"))
    model.start_benchmark("synthetic event", brand_id="competitor_1")
    assert model.focal_brand_id == "brand_1"


def test_structured_post_metadata_survives_normalization_and_deduplication():
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=7)
    message = {
        "post_id": "post_fixed",
        "sender_id": "brand_1",
        "sender_role": "brand",
        "message_type": "official_post",
        "content": {
            "content": "公开说明",
            "brand_id": "brand_1",
            "info_impact": {"capability": 0.2, "value": 0.3, "emotion": 0.1},
            "verification_factor": 0.5,
        },
    }
    model.record_message(message)
    model.record_message(message)
    assert len(model._message_buffer_current) == 1
    payload = model._message_buffer_current[0]["content"]
    assert payload["info_impact"]["value"] == 0.3
    assert payload["verification_factor"] == 0.5


def test_tiny_small_world_network_is_valid():
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=7)
    assert model.network.number_of_nodes() == 2


def test_competitor_pressure_runs_in_the_strategic_phase():
    node = SimulationNode(
        num_consumers=1,
        num_brands=1,
        seed=7,
        hyperparams={"competitor_pressure_mu": 1.0},
        population_spec={"consumers": [], "competitors": [{"agent_id": "competitor_01", "persona": "defender"}]},
    )
    node.enable_agent_llms(model_name=None)
    competitor = node._competitors[0]
    competitor.strategic_decision = lambda brief: {
        "final_action": {"final_action_content": "竞争对手发布一条合成声明。"}
    }
    node._competitor_phase()
    assert any(m.get("sender_role") == "competitor" for m in node.model._message_buffer_prev)


def test_full_offline_round_has_one_canonical_post_per_actor(monkeypatch):
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")
    node = SimulationNode(num_consumers=1, num_brands=1, seed=11)
    node.enable_agent_llms(
        model_name="offline",
        brand_model_name="offline",
        competitor_model_name="offline",
        consumer_model_name="offline",
    )
    for step in (1, 2):
        node.run(
            SandboxState(step=step, background_info={"exam_text": "synthetic event", "dominant_topic": "safety"})
        )
        record = node.model.jsonl_records[-1]
        roles = [post.get("authorRole") for post in record["consumer_generated_content"]]
        assert roles.count("brand") == 1
        assert roles.count("competitor") == 1
        assert roles.count("consumer") == 1
        assert record["timestep"] == step
