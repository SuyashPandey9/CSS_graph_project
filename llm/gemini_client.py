"""Gemini LLM adapter for V3.

Uses the Gemini REST API with the model specified in config.yaml.
Designed for the free-tier gemini-2.5-flash-preview model by default.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config.settings import get_settings


@dataclass(frozen=True)
class GeminiResponse:
    """Normalized response from Gemini."""

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


def _build_request(model: str, api_key: str, prompt: str) -> urllib.request.Request:
    """Build the Gemini generateContent request."""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def list_models(api_key: str, *, timeout: int = 30) -> list[str]:
    """List available Gemini models for the given API key."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
    return models


def generate_text(prompt: str, *, timeout: int = 30, max_retries: int = 3) -> GeminiResponse:
    """Send a prompt to Gemini and return the response text.

    Raises:
        RuntimeError: for API errors such as rate limits.
    """

    settings = get_settings()
    config = _load_config()
    model = config.get("primary_llm_model")
    if not model:
        raise ValueError("primary_llm_model is required in config.yaml.")

    req = _build_request(model=model, api_key=settings.gemini_api_key, prompt=prompt)
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                wait_seconds = None
                if retry_after:
                    try:
                        wait_seconds = int(retry_after)
                    except ValueError:
                        wait_seconds = None
                if wait_seconds is None:
                    wait_seconds = min(2 ** attempt, 8)
                time.sleep(wait_seconds)
                attempt += 1
                continue
            if exc.code == 429:
                raise RuntimeError(
                    "Gemini rate limit exceeded (HTTP 429). Try again later."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    "Gemini model not found (HTTP 404). Update config.yaml to a model "
                    "available for your key."
                ) from exc
            raise RuntimeError(f"Gemini API error (HTTP {exc.code}).") from exc
        except urllib.error.URLError as exc:
            _debug_log("gemini_url_error", {"reason": str(exc.reason)}, "H7")
            raise RuntimeError("Gemini API connection error.") from exc

    candidates = payload.get("candidates", [])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            text = parts[0].get("text", "")

    return GeminiResponse(text=text, raw=payload)
