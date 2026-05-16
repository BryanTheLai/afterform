from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from afterform.config import PipelineConfig
from afterform.providers.llm import (
    LlmImageInput,
    ProviderRequestConverter,
    StructuredLlmRequest,
    call_structured_llm,
    resolved_llm_identity,
    resolved_text_model,
)


class _Response(BaseModel):
    value: str


def test_openai_converter_emits_data_url_for_images():
    converter = ProviderRequestConverter("openai")
    request = StructuredLlmRequest(
        stage_name="layout vision",
        model="gpt-5.4",
        system_instruction="sys",
        images=(
            LlmImageInput(
                data=b"\xff\xd8\xff\xd9",
                mime_type="image/jpeg",
                label="FRAME 0: timestamp_sec=1.23",
            ),
        ),
        response_schema=_Response,
    )

    payload = converter.to_openai_input(request)

    assert payload[0]["role"] == "user"
    assert payload[0]["content"][0]["text"] == "FRAME 0: timestamp_sec=1.23"
    assert payload[0]["content"][1]["type"] == "input_image"
    assert payload[0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


def test_resolved_text_model_falls_back_to_azure_deployment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", llm_provider="azure")

    assert resolved_text_model(cfg) == "gpt-5.4"


def test_resolved_llm_identity_prefers_azure_base_url(monkeypatch):
    monkeypatch.setenv("AZURE_BASE_URL", "https://example.services.ai.azure.com/openai/v1/")
    monkeypatch.delenv("AZURE_ENDPOINT", raising=False)
    cfg = PipelineConfig(
        youtube_url="https://youtu.be/x",
        llm_provider="azure",
        llm_model="gpt-5.4",
    )

    identity = resolved_llm_identity(cfg)

    assert identity == {
        "provider": "azure",
        "model": "gpt-5.4",
        "base_url": "https://example.services.ai.azure.com/openai/v1",
    }


def test_call_structured_llm_uses_gemini_client(monkeypatch):
    seen: dict[str, object] = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            seen["model"] = model
            seen["contents"] = contents
            seen["config"] = config
            parsed = _Response(value="ok")
            return SimpleNamespace(text=parsed.model_dump_json(), parsed=parsed)

    class FakeClient:
        def __init__(self, *, api_key, http_options=None):
            seen["api_key"] = api_key
            seen["http_options"] = http_options
            self.models = FakeModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")
    monkeypatch.setattr("afterform.providers.llm.genai.Client", FakeClient)

    response = call_structured_llm(
        StructuredLlmRequest(
            stage_name="clip selection",
            model="gemini-3-flash-preview",
            system_instruction="sys",
            user_text="hello",
            response_schema=_Response,
            max_output_tokens=1234,
        ),
        provider="gemini",
    )

    assert response.parsed == _Response(value="ok")
    assert seen["api_key"] == "test-gemini-key"
    assert seen["model"] == "gemini-3-flash-preview"
    assert seen["contents"] == "hello"
    assert seen["config"].max_output_tokens == 1234


def test_call_structured_llm_uses_openai_client(monkeypatch):
    seen: dict[str, object] = {}

    class FakeResponses:
        def parse(self, **kwargs):
            seen["parse"] = kwargs
            parsed = _Response(value="ok")
            return SimpleNamespace(output_text=parsed.model_dump_json(), output_parsed=parsed)

    class FakeClient:
        def __init__(self, *, api_key, base_url=None, timeout=None, max_retries=0):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            seen["timeout"] = timeout
            seen["max_retries"] = max_retries
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.openai-proxy.local/v1")
    monkeypatch.setattr("afterform.providers.llm.OpenAI", FakeClient)

    response = call_structured_llm(
        StructuredLlmRequest(
            stage_name="clip selection",
            model="gpt-5.4",
            system_instruction="sys",
            user_text="hello",
            response_schema=_Response,
            timeout_ms=12_000,
            max_output_tokens=4567,
            reasoning_effort="minimal",
            verbosity="low",
        ),
        provider="openai",
    )

    assert response.parsed == _Response(value="ok")
    assert seen["api_key"] == "test-openai-key"
    assert seen["base_url"] == "https://example.openai-proxy.local/v1"
    assert seen["timeout"] == 12.0
    assert seen["max_retries"] == 0
    assert seen["parse"]["model"] == "gpt-5.4"
    assert seen["parse"]["instructions"] == "sys"
    assert seen["parse"]["input"] == "hello"
    assert seen["parse"]["text_format"] is _Response
    assert seen["parse"]["max_output_tokens"] == 4567
    assert seen["parse"]["reasoning"] == {"effort": "minimal"}
    assert seen["parse"]["text"] == {"verbosity": "low"}


def test_call_structured_llm_uses_azure_endpoint(monkeypatch):
    seen: dict[str, object] = {}

    class FakeResponses:
        def parse(self, **kwargs):
            seen["parse"] = kwargs
            parsed = _Response(value="ok")
            return SimpleNamespace(output_text=parsed.model_dump_json(), output_parsed=parsed)

    class FakeClient:
        def __init__(
            self,
            *,
            api_key,
            api_version,
            azure_endpoint=None,
            azure_deployment=None,
            base_url=None,
            timeout=None,
            max_retries=0,
        ):
            seen["api_key"] = api_key
            seen["api_version"] = api_version
            seen["azure_endpoint"] = azure_endpoint
            seen["azure_deployment"] = azure_deployment
            seen["base_url"] = base_url
            seen["timeout"] = timeout
            seen["max_retries"] = max_retries
            self.responses = FakeResponses()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example-resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.delenv("AZURE_BASE_URL", raising=False)
    monkeypatch.setattr("afterform.providers.llm.AzureOpenAI", FakeClient)

    response = call_structured_llm(
        StructuredLlmRequest(
            stage_name="clip selection",
            model="gpt-5.4",
            system_instruction="sys",
            user_text="hello",
            response_schema=_Response,
            timeout_ms=7_500,
            max_output_tokens=8910,
            reasoning_effort="low",
            verbosity="medium",
        ),
        provider="azure",
    )

    assert response.parsed == _Response(value="ok")
    assert seen["api_key"] == "test-azure-key"
    assert seen["api_version"] == "2025-03-01-preview"
    assert seen["azure_endpoint"] == "https://example-resource.openai.azure.com/"
    assert seen["azure_deployment"] == "gpt-5.4"
    assert seen["base_url"] is None
    assert seen["timeout"] == 7.5
    assert seen["max_retries"] == 0
    assert seen["parse"]["max_output_tokens"] == 8910
    assert seen["parse"]["reasoning"] == {"effort": "low"}
    assert seen["parse"]["text"] == {"verbosity": "medium"}


def test_call_structured_llm_uses_azure_base_url(monkeypatch):
    seen: dict[str, object] = {}

    class FakeResponses:
        def parse(self, **kwargs):
            seen["parse"] = kwargs
            parsed = _Response(value="ok")
            return SimpleNamespace(output_text=parsed.model_dump_json(), output_parsed=parsed)

    class FakeClient:
        def __init__(self, *, api_key, base_url=None, timeout=None, max_retries=0):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            seen["timeout"] = timeout
            seen["max_retries"] = max_retries
            self.responses = FakeResponses()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_BASE_URL", "https://example.services.ai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://ignored.example.com/")
    monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
    monkeypatch.setattr("afterform.providers.llm.OpenAI", FakeClient)

    response = call_structured_llm(
        StructuredLlmRequest(
            stage_name="clip selection",
            model="gpt-5.4",
            system_instruction="sys",
            user_text="hello",
            response_schema=_Response,
        ),
        provider="azure",
    )

    assert response.parsed == _Response(value="ok")
    assert seen["api_key"] == "test-azure-key"
    assert seen["base_url"] == "https://example.services.ai.azure.com/openai/v1/"
    assert seen["timeout"] is None
    assert seen["max_retries"] == 0

