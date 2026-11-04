"""Per-model overrides from models.yaml `llm.models`."""

from ttrpg_crew.config_loader import config_section
from ttrpg_crew.main import model_params


def test_gpt_oss_gets_low_reasoning_and_gemma_keeps_the_default():
    cfg = config_section("llm")
    base = dict(cfg["defaults"])
    oss = model_params("ollama_chat/gpt-oss:20b", base, cfg.get("models"))
    gemma = model_params("ollama_chat/gemma3:27b", base, cfg.get("models"))
    assert oss["extra_body"] == {"reasoning_effort": "low"}
    assert gemma["extra_body"] == base["extra_body"]
    assert oss["temperature"] == base["temperature"]


def test_no_overrides_leaves_params_unchanged():
    assert model_params("anything", {"temperature": 0.5}, None) == {"temperature": 0.5}
