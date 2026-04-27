"""Provider-agnostic structured LLM calls for stages 2/2.25/2.5/3."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypeVar

from google import genai
from google.genai import types as gemini_types
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from afterform.config import (
    GEMINI_MODEL,
    GEMINI_VISION_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_VISION_MODEL,
    PipelineConfig,
)
from afterform.env import (
    resolve_azure_openai_api_key,
    resolve_azure_openai_api_version,
    resolve_azure_openai_base_url,
    resolve_azure_openai_deployment,
    resolve_azure_openai_endpoint,
    resolve_gemini_api_key,
    resolve_openai_api_key,
    resolve_openai_base_url,
)
from afterform.providers.gemini import gemini_generate_config

ProviderName = Literal["gemini", "openai", "azure"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)

_GEMINI_RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]
_GEMINI_RETRY_INITIAL_DELAY_SEC = 1.0
_GEMINI_RETRY_MAX_DELAY_SEC = 4.0
_GEMINI_RETRY_EXP_BASE = 2.0
_GEMINI_RETRY_JITTER = 0.0


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def normalize_llm_provider(provider: str | None) -> ProviderName:
    value = (provider or LLM_PROVIDER).strip().lower().replace("_", "-")
    aliases: dict[str, ProviderName] = {
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "openai": "openai",
        "azure": "azure",
        "azure-openai": "azure",
    }
    resolved = aliases.get(value)
    if resolved is None:
        raise ValueError(
            f"Unknown AFTERFORM_LLM_PROVIDER={provider!r}. Use one of: gemini, openai, azure."
        )
    return resolved


def resolved_llm_provider(config: PipelineConfig | None = None) -> ProviderName:
    return normalize_llm_provider(config.llm_provider if config is not None else None)


def resolved_text_model(
    config: PipelineConfig | None = None,
    *,
    model_override: str | None = None,
) -> str:
    provider = resolved_llm_provider(config)
    for candidate in (
        model_override,
        config.llm_model if config is not None else None,
        config.gemini_model if config is not None else None,
        LLM_MODEL,
    ):
        value = _clean(candidate)
        if value:
            return value
    if provider == "azure":
        deployment = _clean(resolve_azure_openai_deployment())
        if deployment:
            return deployment
    if provider == "gemini":
        return GEMINI_MODEL
    raise ValueError(
        "Set AFTERFORM_LLM_MODEL or pass --llm-model when AFTERFORM_LLM_PROVIDER "
        "is openai or azure."
    )


def resolved_vision_model(
    config: PipelineConfig | None = None,
    *,
    model_override: str | None = None,
) -> str:
    for candidate in (
        model_override,
        config.llm_vision_model if config is not None else None,
        config.gemini_vision_model if config is not None else None,
        LLM_VISION_MODEL,
        GEMINI_VISION_MODEL,
    ):
        value = _clean(candidate)
        if value:
            return value
    return resolved_text_model(config)


def resolved_llm_identity(
    config: PipelineConfig,
    *,
    vision: bool = False,
) -> dict[str, str]:
    provider = resolved_llm_provider(config)
    model = resolved_vision_model(config) if vision else resolved_text_model(config)
    identity: dict[str, str] = {"provider": provider, "model": model}
    if provider == "openai":
        base_url = _clean(resolve_openai_base_url())
        if base_url:
            identity["base_url"] = base_url.rstrip("/")
        return identity
    if provider == "azure":
        endpoint = _clean(resolve_azure_openai_endpoint())
        base_url = _clean(resolve_azure_openai_base_url())
        api_version = _clean(resolve_azure_openai_api_version())
        deployment = _clean(resolve_azure_openai_deployment())
        if base_url:
            endpoint = None
        if not endpoint and not base_url:
            raise ValueError(
                "Azure provider requires AZURE_OPENAI_ENDPOINT/AZURE_ENDPOINT or "
                "AZURE_OPENAI_BASE_URL/AZURE_BASE_URL."
            )
        if base_url:
            identity["base_url"] = base_url.rstrip("/")
            return identity
        if not api_version:
            raise ValueError(
                "Azure endpoint transport requires AZURE_OPENAI_API_VERSION or OPENAI_API_VERSION."
            )
        if endpoint:
            identity["azure_endpoint"] = endpoint.rstrip("/")
        if endpoint and deployment:
            identity["azure_deployment"] = deployment
        identity["api_version"] = api_version
        return identity
    return identity


@dataclass(frozen=True)
class LlmImageInput:
    path: str | Path | None = None
    data: bytes | None = None
    mime_type: str = "image/jpeg"
    label: str = ""
    detail: str = "high"

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is None:
            raise ValueError("LlmImageInput requires either path or data.")
        return Path(self.path).read_bytes()


@dataclass(frozen=True)
class StructuredLlmRequest(Generic[SchemaT]):
    stage_name: str
    model: str
    system_instruction: str
    user_text: str = ""
    response_schema: type[SchemaT] | None = None
    temperature: float = 0.2
    images: tuple[LlmImageInput, ...] = field(default_factory=tuple)
    timeout_ms: int | None = None
    max_retries: int = 0


@dataclass(frozen=True)
class StructuredLlmResponse(Generic[SchemaT]):
    raw_text: str
    parsed: SchemaT | None


class ProviderRequestConverter:
    """Convert a provider-agnostic request into provider-specific payloads."""

    def __init__(self, provider: ProviderName):
        self.provider = normalize_llm_provider(provider)

    def to_gemini_contents(self, request: StructuredLlmRequest[SchemaT]):
        if not request.images:
            return request.user_text
        contents: list[object] = []
        if request.user_text:
            contents.append(gemini_types.Part.from_text(text=request.user_text))
        for image in request.images:
            if image.label:
                contents.append(gemini_types.Part.from_text(text=image.label))
            contents.append(
                gemini_types.Part.from_bytes(
                    data=image.read_bytes(),
                    mime_type=image.mime_type,
                )
            )
        return contents

    def to_openai_input(self, request: StructuredLlmRequest[SchemaT]):
        if not request.images:
            return request.user_text
        content: list[dict[str, str]] = []
        if request.user_text:
            content.append({"type": "input_text", "text": request.user_text})
        for image in request.images:
            if image.label:
                content.append({"type": "input_text", "text": image.label})
            b64 = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image.mime_type};base64,{b64}",
                    "detail": image.detail,
                }
            )
        return [{"role": "user", "content": content}]

    def parse_gemini_response(
        self,
        response: object,
        schema: type[SchemaT] | None,
    ) -> StructuredLlmResponse[SchemaT]:
        parsed = None
        raw = getattr(response, "text", None) or ""
        if schema is not None and isinstance(getattr(response, "parsed", None), schema):
            parsed = getattr(response, "parsed")
        if not raw and parsed is None:
            raise RuntimeError("Gemini returned neither text nor parsed response.")
        if parsed is None and schema is not None and raw:
            parsed = schema.model_validate_json(raw)
        if not raw and parsed is not None:
            raw = parsed.model_dump_json()
        return StructuredLlmResponse(raw_text=raw, parsed=parsed)

    def parse_openai_response(
        self,
        response: object,
        schema: type[SchemaT] | None,
    ) -> StructuredLlmResponse[SchemaT]:
        parsed = None
        raw = getattr(response, "output_text", "") or ""
        output_parsed = getattr(response, "output_parsed", None)
        if schema is not None and isinstance(output_parsed, schema):
            parsed = output_parsed
        if not raw and parsed is None:
            raise RuntimeError("OpenAI returned neither output_text nor parsed response.")
        if parsed is None and schema is not None and raw:
            parsed = schema.model_validate_json(raw)
        if not raw and parsed is not None:
            raw = parsed.model_dump_json()
        return StructuredLlmResponse(raw_text=raw, parsed=parsed)


def _build_gemini_client(*, timeout_ms: int | None, max_retries: int) -> genai.Client:
    kwargs: dict[str, object] = {"api_key": resolve_gemini_api_key()}
    http_kwargs: dict[str, object] = {}
    if timeout_ms is not None:
        http_kwargs["timeout"] = timeout_ms
    if max_retries > 0:
        http_kwargs["retryOptions"] = gemini_types.HttpRetryOptions(
            attempts=max_retries,
            initialDelay=_GEMINI_RETRY_INITIAL_DELAY_SEC,
            maxDelay=_GEMINI_RETRY_MAX_DELAY_SEC,
            expBase=_GEMINI_RETRY_EXP_BASE,
            jitter=_GEMINI_RETRY_JITTER,
            httpStatusCodes=_GEMINI_RETRYABLE_STATUS_CODES,
        )
    if http_kwargs:
        kwargs["http_options"] = gemini_types.HttpOptions(**http_kwargs)
    return genai.Client(**kwargs)


def _build_openai_client(*, timeout_ms: int | None, max_retries: int) -> OpenAI:
    kwargs: dict[str, object] = {
        "api_key": resolve_openai_api_key(),
        "max_retries": max(0, max_retries),
    }
    base_url = _clean(resolve_openai_base_url())
    if base_url:
        kwargs["base_url"] = base_url
    if timeout_ms is not None:
        kwargs["timeout"] = timeout_ms / 1000.0
    return OpenAI(**kwargs)


def _build_azure_openai_client(*, timeout_ms: int | None, max_retries: int):
    endpoint = _clean(resolve_azure_openai_endpoint())
    base_url = _clean(resolve_azure_openai_base_url())
    api_version = _clean(resolve_azure_openai_api_version())
    deployment = _clean(resolve_azure_openai_deployment())
    if base_url:
        endpoint = None
    if not endpoint and not base_url:
        raise ValueError(
            "Azure provider requires AZURE_OPENAI_ENDPOINT/AZURE_ENDPOINT or "
            "AZURE_OPENAI_BASE_URL/AZURE_BASE_URL."
        )
    common_kwargs: dict[str, object] = {
        "api_key": resolve_azure_openai_api_key(),
        "max_retries": max(0, max_retries),
    }
    if timeout_ms is not None:
        common_kwargs["timeout"] = timeout_ms / 1000.0
    if base_url:
        return OpenAI(base_url=base_url, **common_kwargs)
    if not api_version:
        raise ValueError(
            "Azure endpoint transport requires AZURE_OPENAI_API_VERSION or OPENAI_API_VERSION."
        )
    kwargs = dict(common_kwargs)
    kwargs["api_version"] = api_version
    if endpoint:
        kwargs["azure_endpoint"] = endpoint
    if endpoint and deployment:
        kwargs["azure_deployment"] = deployment
    return AzureOpenAI(**kwargs)


def call_structured_llm(
    request: StructuredLlmRequest[SchemaT],
    *,
    provider: ProviderName | str,
) -> StructuredLlmResponse[SchemaT]:
    provider_name = normalize_llm_provider(provider)
    converter = ProviderRequestConverter(provider_name)
    if provider_name == "gemini":
        client = _build_gemini_client(
            timeout_ms=request.timeout_ms,
            max_retries=request.max_retries,
        )
        response = client.models.generate_content(
            model=request.model,
            contents=converter.to_gemini_contents(request),
            config=gemini_generate_config(
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                response_mime_type="application/json",
                response_schema=request.response_schema,
            ),
        )
        return converter.parse_gemini_response(response, request.response_schema)
    if provider_name == "openai":
        client = _build_openai_client(
            timeout_ms=request.timeout_ms,
            max_retries=request.max_retries,
        )
    else:
        client = _build_azure_openai_client(
            timeout_ms=request.timeout_ms,
            max_retries=request.max_retries,
        )
    response = client.responses.parse(
        model=request.model,
        instructions=request.system_instruction,
        input=converter.to_openai_input(request),
        temperature=request.temperature,
        text_format=request.response_schema,
    )
    return converter.parse_openai_response(response, request.response_schema)

