"""Phase 0 verification script for configuration loading and Gemini/Groq checks."""

from __future__ import annotations

from config.settings import get_settings

import json
import urllib.error
import urllib.request


def _ping_gemini(api_key: str) -> None:
    """List Gemini models to verify the key is valid."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_count = len(payload.get("models", []))
    print(f"Gemini API ping succeeded. Models returned: {model_count}")


def _ping_groq(api_key: str) -> None:
    """List Groq models to verify the key is valid."""

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_count = len(payload.get("data", []))
        print(f"Groq API ping succeeded. Models returned: {model_count}")
    except urllib.error.HTTPError as exc:
        print(
            f"Groq API ping failed with HTTP {exc.code} ({exc.reason}). "
            "Verify GROQ_API_KEY or account status."
        )


def main() -> None:
    """Load settings and verify Gemini/Groq API keys."""

    settings = get_settings()
    key_preview = f"{settings.gemini_api_key[:4]}...{settings.gemini_api_key[-4:]}"
    print("Config loaded successfully.")
    print(f"GEMINI_API_KEY detected: {key_preview}")

    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is required for Phase 0 verification. "
            "Add it to .env and re-run test_config.py."
        )

    _ping_gemini(settings.gemini_api_key)
    _ping_groq(settings.groq_api_key)


if __name__ == "__main__":
    main()
