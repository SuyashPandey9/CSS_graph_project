"""Configuration loader for V3.

Loads environment variables from a local .env file and exposes a settings object.
No secrets are hardcoded; values are sourced from the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Strongly-typed settings loaded from environment variables."""

    gemini_api_key: str
    openai_api_key: str | None
    groq_api_key: str | None


def _load_env() -> None:
    """Load a local .env file if present, without overriding existing env vars."""

    root = Path(__file__).resolve().parent.parent
    candidates = [root / ".env", root / ".env.txt"]
    env_path = next((p for p in candidates if p.exists()), candidates[0])
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


def get_settings() -> Settings:
    """Load and validate settings from the environment."""

    _load_env()
    import os

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None

    # At least one LLM API key is required
    if not gemini_api_key and not openai_api_key:
        raise ValueError(
            "At least one LLM API key is required (GEMINI_API_KEY or OPENAI_API_KEY). "
            "Copy .env.example to .env and set it."
        )

    return Settings(
        gemini_api_key=gemini_api_key or "",
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
    )
