"""
Multi-provider LLM client with structured JSON parsing.

Routes calls based on model name prefix:
  - claude-*  -> Anthropic API
  - gemini-*  -> Google GenAI SDK
  - gpt-* / o1-* / o3-* / o4-*  -> OpenAI API

All LLM calls go through query_llm(). It handles retries, JSON extraction,
and provider routing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# -- Lazy-init clients ------------------------------------------------------

_anthropic_client = None
_gemini_client = None
_openai_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise RuntimeError("Set ANTHROPIC_API_KEY in .env or environment.")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise RuntimeError("Set GEMINI_API_KEY in .env or environment.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise RuntimeError("Set OPENAI_API_KEY in .env or environment.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# -- JSON extraction --------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """Pull JSON from an LLM response that may contain markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


# -- Provider-specific calls ------------------------------------------------

def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Anthropic API and return raw text."""
    client = _get_anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Google GenAI SDK and return raw text."""
    from google.genai import types

    client = _get_gemini()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call OpenAI API and return raw text."""
    client = _get_openai()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _is_openai_model(model: str) -> bool:
    """Check if model name belongs to OpenAI."""
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


def _call_provider(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Route to the right provider based on model name."""
    if model.startswith("gemini"):
        return _call_gemini(system_prompt, user_prompt, model, max_tokens, temperature)
    elif _is_openai_model(model):
        return _call_openai(system_prompt, user_prompt, model, max_tokens, temperature)
    else:
        return _call_anthropic(system_prompt, user_prompt, model, max_tokens, temperature)


# -- Public API -------------------------------------------------------------

def query_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    retries: int = 5,
) -> Dict[str, Any]:
    """
    Send a prompt pair to an LLM and return parsed JSON.

    Routes to Anthropic or Gemini based on model prefix.
    Retries on transient failures and JSON parse errors.
    """
    last_error: Optional[Exception] = None
    max_attempts = retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = _call_provider(
                system_prompt, user_prompt, model, max_tokens, temperature
            )
            parsed = _extract_json(raw_text)
            log.debug("LLM call succeeded on attempt %d (model=%s)", attempt, model)
            return parsed

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            log.warning(
                "JSON parse failed (attempt %d/%d, model=%s): %s",
                attempt, max_attempts, model, exc,
            )
        except Exception as exc:
            last_error = exc
            is_overloaded = "529" in str(exc) or "overloaded" in str(exc).lower()
            log.warning(
                "LLM call failed (attempt %d/%d, model=%s): %s",
                attempt, max_attempts, model, exc,
            )
            # On overload errors, use longer backoff to let the API recover
            if is_overloaded and attempt < max_attempts:
                wait = min(10 * attempt, 60)
                log.info("Overloaded -- waiting %ds before retry...", wait)
                time.sleep(wait)
                continue

        if attempt < max_attempts:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"LLM call failed after {max_attempts} attempts (model={model}). "
        f"Last error: {last_error}"
    )
