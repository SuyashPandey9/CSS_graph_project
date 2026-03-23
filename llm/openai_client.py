"""OpenAI LLM adapter for V3.

Uses the OpenAI REST API with configurable model.
Drop-in replacement for gemini_client with identical interface.
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
class LLMResponse:
    """Normalized response from LLM."""

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
    """Build the OpenAI chat completions request."""

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )


def generate_text(prompt: str, *, timeout: int = 60, max_retries: int = 3) -> LLMResponse:
    """Send a prompt to OpenAI and return the response text.

    Raises:
        RuntimeError: for API errors such as rate limits.
    """

    settings = get_settings()
    config = _load_config()
    model = config.get("openai_model", "gpt-3.5-turbo")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required in .env when using OpenAI provider.")

    req = _build_request(model=model, api_key=settings.openai_api_key, prompt=prompt)
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
                    "OpenAI rate limit exceeded (HTTP 429). Try again later."
                ) from exc
            if exc.code == 401:
                raise RuntimeError(
                    "OpenAI authentication failed (HTTP 401). Check your API key."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    "OpenAI model not found (HTTP 404). Update openai_model in config.yaml."
                ) from exc
            raise RuntimeError(f"OpenAI API error (HTTP {exc.code}).") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("OpenAI API connection error.") from exc

    # Extract text from OpenAI response format
    choices = payload.get("choices", [])
    text = ""
    if choices:
        message = choices[0].get("message", {})
        text = message.get("content", "")

    return LLMResponse(text=text, raw=payload)
