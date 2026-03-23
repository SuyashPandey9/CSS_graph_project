"""Phase 1 verification script for the Gemini LLM adapter."""

from __future__ import annotations

from llm.llm_client import generate_text


def main() -> None:
    """Send a test prompt to Gemini and print the response."""

    prompt = "In one sentence, explain what solar energy is."
    response = generate_text(prompt)
    print("Gemini response:")
    print(response.text.strip() or "(empty response)")


if __name__ == "__main__":
    main()
