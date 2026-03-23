"""Unified LLM client for V3.

Routes requests to the appropriate backend (Gemini or OpenAI) based on config.yaml.
All modules should import generate_text from this file instead of the individual clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from any LLM backend."""

    text: str
    raw: dict


def _load_config() -> dict:
    """Load simple key:value pairs from config.yaml."""

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found at project root.")

    config: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def generate_text(prompt: str, *, timeout: int = 60, max_retries: int = 3) -> LLMResponse:
    """Send a prompt to the configured LLM backend and return the response.

    The backend is determined by 'llm_provider' in config.yaml:
      - 'openai' -> uses OpenAI API
      - 'gemini' -> uses Gemini API (default)

    Raises:
        RuntimeError: for API errors such as rate limits.
        ValueError: for missing API keys or invalid configuration.
    """

    config = _load_config()
    provider = config.get("llm_provider", "gemini").lower()

    if provider == "openai":
        from llm.openai_client import generate_text as openai_generate
        from llm.openai_client import LLMResponse as OpenAIResponse
        result = openai_generate(prompt, timeout=timeout, max_retries=max_retries)
        return LLMResponse(text=result.text, raw=result.raw)

    elif provider == "gemini":
        from llm.gemini_client import generate_text as gemini_generate
        result = gemini_generate(prompt, timeout=timeout, max_retries=max_retries)
        return LLMResponse(text=result.text, raw=result.raw)

    else:
        raise ValueError(
            f"Unknown llm_provider '{provider}' in config.yaml. "
            "Use 'openai' or 'gemini'."
        )
