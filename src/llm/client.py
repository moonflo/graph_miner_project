"""OpenAI-compatible Chat Completions client for LLM extraction."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class MissingLLMConfigError(RuntimeError):
    """Raised when a real LLM call is requested without required settings."""


@dataclass(frozen=True)
class LLMClientConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 1.0


class LLMClient:
    """Small OpenAI-compatible Chat Completions client.

    The client reads secrets from environment variables or a local ``.env`` file
    via ``from_env``. API keys are never logged or included in raised messages.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        self.config = LLMClientConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    @property
    def model(self) -> str:
        return self.config.model

    @classmethod
    def from_env(
        cls,
        *,
        model_override: str | None = None,
        dotenv_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "LLMClient":
        """Build a client from environment variables or a ``.env`` file."""

        if env is None:
            load_dotenv(dotenv_path)
            env_values: Mapping[str, str] = os.environ
        else:
            env_values = env

        api_key = _env_str(env_values, "LLM_API_KEY")
        base_url = _env_str(env_values, "LLM_BASE_URL")
        model = model_override or _env_str(env_values, "LLM_MODEL")

        missing = []
        if not api_key:
            missing.append("LLM_API_KEY")
        if not base_url:
            missing.append("LLM_BASE_URL")
        if not model:
            missing.append("LLM_MODEL")
        if missing:
            raise MissingLLMConfigError(
                "Missing required LLM configuration: "
                + ", ".join(missing)
                + ". Set these in the environment or a .env file, or use --mock/--dry-run."
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=_env_float(env_values, "LLM_TEMPERATURE", 0.0),
            max_tokens=_env_int(env_values, "LLM_MAX_TOKENS", 4096),
            timeout=_env_float(env_values, "LLM_TIMEOUT", 60.0),
            max_retries=_env_int(env_values, "LLM_MAX_RETRIES", 2),
            retry_backoff=_env_float(env_values, "LLM_RETRY_BACKOFF", 1.0),
        )

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Call the configured Chat Completions endpoint and return text."""

        client = self._build_openai_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:  # pragma: no cover - depends on remote API.
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(self.config.retry_backoff * (2**attempt))

        error_name = type(last_error).__name__ if last_error else "UnknownError"
        raise RuntimeError(
            f"LLM API request failed after {attempts} attempt(s): {error_name}"
        ) from last_error

    def _build_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - covered by docs/requirements.
            raise RuntimeError(
                "The openai package is required for real LLM extraction. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        return OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)


class MockLLMClient:
    """Deterministic no-network client for smoke tests and local plumbing."""

    model = "mock-llm"

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del prompt, system_prompt
        return json.dumps(
            {
                "entities": [
                    {
                        "name": "OpenAI",
                        "type": "organization",
                        "aliases": [],
                        "evidence": "OpenAI released ChatGPT",
                    },
                    {
                        "name": "ChatGPT",
                        "type": "concept",
                        "aliases": [],
                        "evidence": "OpenAI released ChatGPT",
                    },
                ],
                "relations": [
                    {
                        "head": "OpenAI",
                        "head_type": "organization",
                        "tail": "ChatGPT",
                        "tail_type": "concept",
                        "relation_type": "released",
                        "evidence": "OpenAI released ChatGPT",
                        "confidence": 1.0,
                    }
                ],
                "triples": [
                    {
                        "subject": "OpenAI",
                        "predicate": "released",
                        "object": "ChatGPT",
                        "evidence": "OpenAI released ChatGPT",
                    }
                ],
            },
            ensure_ascii=False,
        )


def load_dotenv(dotenv_path: str | Path | None = None) -> None:
    """Load environment variables from python-dotenv or a simple fallback parser."""

    path = Path(dotenv_path) if dotenv_path is not None else Path(".env")
    try:
        from dotenv import load_dotenv as load_dotenv_package

        load_dotenv_package(path if path.is_file() else None, override=False)
        return
    except ImportError:
        pass

    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(env: Mapping[str, str], key: str) -> str:
    return str(env.get(key, "")).strip()


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = _env_str(env, key)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise MissingLLMConfigError(f"{key} must be an integer.") from exc


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = _env_str(env, key)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise MissingLLMConfigError(f"{key} must be a number.") from exc
