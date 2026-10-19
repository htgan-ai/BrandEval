"""Runtime regressions for consumer message and offline behavior."""

from brandeval.model import BrandEvalModel
from brandeval.agents.consumer import Consumer


def _model_and_consumer():
    model = BrandEvalModel(num_consumers=1, num_brands=1, seed=17)
    consumer = list(model.agents_by_type["consumer"])[0]
    focal = list(model.agents_by_type["brand"])[0].unique_id
    return model, consumer, focal


def test_structured_post_metadata_reaches_cbbe_verification_gate():
    model, consumer, focal = _model_and_consumer()
    before = dict(consumer.cbbe_state[focal])
    consumer.receive_message({
        "message_type": "official_post",
        "sender_role": "brand",
        "content": {
            "content": "可核验说明",
            "brand_id": focal,
            "info_impact": {"capability": 0.4, "value": 0.2, "emotion": 0.1},
            "verification_factor": 0.25,
        },
    })
    consumer._process_messages()
    assert consumer.last_verification_status[focal] == 0.25
    assert consumer.cbbe_state[focal] != before


def test_ugc_is_recorded_once_with_author_and_topic_metadata(monkeypatch):
    model, consumer, focal = _model_and_consumer()
    model.topic_keywords = ["安全"]
    consumer.llm_model_name = "offline"
    consumer.stance_by_brand[focal] = "opponent"
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")

    consumer._generate_ugc_for_brand(focal, seed_text="安全提示")

    assert len(model._message_buffer_current) == 1
    assert len(model._posts_current) == 1
    message = model._message_buffer_current[0]
    payload = message["content"]
    assert payload["brand_id"] == focal
    assert payload["topicTags"] == ["安全"]
    assert payload["authorPersonality"]["initial_stance"] == consumer.current_stance_global
    assert "质疑安全与透明" in payload["content"]


def test_benchmark_generates_only_one_focal_post(monkeypatch):
    model, consumer, focal = _model_and_consumer()
    consumer.llm_model_name = "offline"
    monkeypatch.setenv("BRANDEVAL_API_DISABLE_NETWORK", "1")

    model.start_benchmark("初始事件", brand_id=focal)
    assert model.focal_brand_id == focal
    # t=0 is committed before the model flips to the next-step buffer.
    posts = model.jsonl_records[-1]["consumer_generated_content"]
    assert len(posts) == 1
    assert {post["brand_id"] for post in posts} == {focal}
    assert len(model._message_buffer_prev) == 1


def test_neighbor_placement_is_stable_across_model_instances():
    first = _model_and_consumer()[1]
    second = _model_and_consumer()[1]
    first.get_neighbors()
    second.get_neighbors()
    assert first.pos == second.pos
