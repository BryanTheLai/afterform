"""Tests for clip selection env handling and provider plumbing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_resolve_gemini_api_key_prefers_google_over_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
    from afterform.env import resolve_gemini_api_key

    assert resolve_gemini_api_key() == "from-google"


def test_resolve_gemini_api_key_falls_back_to_gemini(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "only-gemini")
    from afterform.env import resolve_gemini_api_key

    assert resolve_gemini_api_key() == "only-gemini"


def test_resolve_gemini_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "  key  ")
    from afterform.env import resolve_gemini_api_key

    assert resolve_gemini_api_key() == "key"


def test_resolve_gemini_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from afterform.env import resolve_gemini_api_key

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        resolve_gemini_api_key()


@patch("afterform.flows.long_to_shorts.select_clips.call_structured_llm")
def test_select_clips_uses_provider_layer(mock_call):
    from afterform.flows.long_to_shorts.select_clips import ClipSelectionResponse, select_clips

    mock_call.return_value = SimpleNamespace(
        raw_text='{"clips": []}',
        parsed=ClipSelectionResponse(clips=[]),
    )

    select_clips({"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]})

    mock_call.assert_called_once()
    request = mock_call.call_args.args[0]
    assert request.model
    assert request.stage_name == "clip selection"
    assert "hi" in request.user_text
    assert mock_call.call_args.kwargs["provider"] == "gemini"


def test_select_clips_raises_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    from afterform.flows.long_to_shorts.select_clips import select_clips

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        select_clips({"segments": []})

