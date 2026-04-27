"""Environment bootstrap (``.env``) and cache path helpers."""

from __future__ import annotations

import os
from pathlib import Path

_BOOTSTRAPPED = False


def bootstrap_env() -> None:
    """Load ``.env`` from the process cwd (non-fatal if missing). Safe to call twice."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    _BOOTSTRAPPED = True


def _read_first_env(*env_names: str) -> str | None:
    bootstrap_env()
    for env_name in env_names:
        val = (os.environ.get(env_name) or "").strip()
        if val:
            return val
    return None


def default_afterform_cache_root() -> Path:
    """Default cache root for Afterform."""
    override = _read_first_env("AFTERFORM_CACHE_ROOT")
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "afterform"
    return Path.home() / ".cache" / "afterform"


def resolve_gemini_api_key() -> str:
    """Return an API key for Gemini, or raise if none is configured.

    Prefer ``GOOGLE_API_KEY``; fall back to ``GEMINI_API_KEY``. Values are read from
    the environment after ``bootstrap_env()`` (``.env`` in cwd).

    We require an explicit key so we do not fall back to Application Default
    Credentials (e.g. ``gcloud auth application-default login``), which often
    lack the Generative Language API scope and produce
    ``403 ACCESS_TOKEN_SCOPE_INSUFFICIENT``.
    """
    val = _read_first_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if val:
        return val
    raise ValueError(
        "Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini clip selection. "
        "See docs/ENVIRONMENT.md. Without an API key the client may use ADC and fail "
        "with insufficient scopes (403)."
    )


def resolve_openai_api_key() -> str:
    val = _read_first_env("OPENAI_API_KEY")
    if val:
        return val
    raise ValueError("Set OPENAI_API_KEY for OpenAI Responses API stages.")


def resolve_openai_base_url() -> str | None:
    return _read_first_env("OPENAI_BASE_URL")


def resolve_azure_openai_api_key() -> str:
    val = _read_first_env("AZURE_OPENAI_API_KEY")
    if val:
        return val
    raise ValueError("Set AZURE_OPENAI_API_KEY for Azure OpenAI Responses API stages.")


def resolve_azure_openai_endpoint() -> str | None:
    return _read_first_env("AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT")


def resolve_azure_openai_base_url() -> str | None:
    return _read_first_env("AZURE_OPENAI_BASE_URL", "AZURE_BASE_URL")


def resolve_azure_openai_api_version() -> str | None:
    return _read_first_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION")


def resolve_azure_openai_deployment() -> str | None:
    return _read_first_env("AZURE_OPENAI_DEPLOYMENT", "AZURE_DEPLOYMENT")

