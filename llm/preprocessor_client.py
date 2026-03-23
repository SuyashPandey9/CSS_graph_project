"""LLM client adapter for the query preprocessor.

Fix C1: Provides the .call(prompt, system, temperature) interface
expected by tools.query_preprocessor.preprocess_query().

Uses the same OpenAI configuration as the main LLM client (config.yaml)
so that both V3 generation and preprocessing use the same model.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
import time
from pathlib import Path

from config.settings import get_settings


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


class PreprocessorLLMClient:
    """Thin wrapper providing the .call() interface for query_preprocessor.
    
    Usage:
        client = PreprocessorLLMClient()
        response_text = client.call(prompt="...", system="...", temperature=0)
    """

    def __init__(self) -> None:
        settings = get_settings()
        config = _load_config()
        
        provider = config.get("llm_provider", "gemini").lower()
        
        if provider == "openai":
            self._api_key = settings.openai_api_key
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY required for preprocessor LLM client.")
            self._model = config.get("openai_model", "gpt-4o-mini")
            self._provider = "openai"
        elif provider == "gemini":
            self._api_key = settings.gemini_api_key
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY required for preprocessor LLM client.")
            self._model = config.get("primary_llm_model", "gemini-2.0-flash")
            self._provider = "gemini"
        else:
            raise ValueError(f"Unknown llm_provider '{provider}' in config.yaml.")

    def call(self, prompt: str, system: str = "", temperature: float = 0) -> str:
        """Send a prompt to the LLM and return the response text.
        
        Args:
            prompt: The user message content
            system: System prompt / instructions
            temperature: Sampling temperature (0 = deterministic)
        
        Returns:
            Raw response text from the LLM
        """
        if self._provider == "openai":
            return self._call_openai(prompt, system, temperature)
        elif self._provider == "gemini":
            return self._call_gemini(prompt, system, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

    def _call_openai(self, prompt: str, system: str, temperature: float) -> str:
        """Call OpenAI chat completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 500,  # Preprocessor output is small JSON
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    wait = min(2 ** attempt, 8)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenAI API error (HTTP {exc.code}).") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError("OpenAI API connection error.") from exc
        
        return ""

    def _call_gemini(self, prompt: str, system: str, temperature: float) -> str:
        """Call Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        
        # Combine system + user prompt for Gemini
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 500,
            },
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    wait = min(2 ** attempt, 8)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API error (HTTP {exc.code}).") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError("Gemini API connection error.") from exc
        
        return ""


# Cached singleton instance
_CLIENT: PreprocessorLLMClient | None = None


def get_preprocessor_client() -> PreprocessorLLMClient:
    """Get or create the cached preprocessor LLM client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = PreprocessorLLMClient()
    return _CLIENT
