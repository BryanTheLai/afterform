"""Shared Gemini config (thinking) â€” no live API."""

from afterform.providers.gemini import gemini_generate_config


def test_gemini_generate_config_merges_thinking_with_call_site_fields():
    cfg = gemini_generate_config(
        temperature=0.2,
        response_mime_type="application/json",
        system_instruction="x",
    )
    assert cfg.temperature == 0.2
    assert cfg.response_mime_type == "application/json"
    assert cfg.system_instruction == "x"
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 1024
    assert cfg.thinking_config.include_thoughts is True

