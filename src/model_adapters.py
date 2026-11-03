"""Provider adapters that share a single generation interface."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AdapterError(RuntimeError):
    """A provider call failed in a way that can be checkpointed and retried."""


@dataclass
class GenerationResponse:
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, **settings: Any) -> GenerationResponse:
        """Return raw text and available generation metadata."""


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise AdapterError(f"Provider returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise AdapterError(f"Provider request failed: {error}") from error
    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise AdapterError("Provider returned invalid JSON") from error
    if not isinstance(result, dict):
        raise AdapterError("Provider returned an unexpected JSON response")
    return result


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        api_key_env: str,
        endpoint: str,
        timeout: float = 90,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, **settings: Any) -> GenerationResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AdapterError(f"Required API key environment variable is unset: {self.api_key_env}")
        result = _post_json(
            f"{self.endpoint}/chat/completions",
            {"Authorization": f"Bearer {api_key}"},
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": settings.get("temperature", 0.2),
                "max_tokens": settings.get("max_tokens", 300),
            },
            float(settings.get("request_timeout_seconds", self.timeout)),
        )
        try:
            raw_text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise AdapterError("Provider response is missing message content") from error
        if not isinstance(raw_text, str):
            raise AdapterError("Provider message content is not plain text")
        metadata = {
            key: result[key]
            for key in ("id", "model", "usage", "system_fingerprint")
            if key in result
        }
        return GenerationResponse(raw_text=raw_text, metadata=metadata)


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        api_key_env: str,
        endpoint: str,
        api_version: str,
        timeout: float = 90,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout

    def generate(self, prompt: str, **settings: Any) -> GenerationResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AdapterError(f"Required API key environment variable is unset: {self.api_key_env}")
        result = _post_json(
            f"{self.endpoint}/messages",
            {"x-api-key": api_key, "anthropic-version": self.api_version},
            {
                "model": self.model,
                "max_tokens": settings.get("max_tokens", 300),
                "temperature": settings.get("temperature", 0.2),
                "messages": [{"role": "user", "content": prompt}],
            },
            float(settings.get("request_timeout_seconds", self.timeout)),
        )
        try:
            content = result["content"]
            raw_text = "\n".join(
                block["text"] for block in content if block.get("type") == "text"
            )
        except (KeyError, TypeError) as error:
            raise AdapterError("Provider response is missing text content") from error
        if not raw_text:
            raise AdapterError("Provider returned no text content")
        metadata = {
            key: result[key]
            for key in ("id", "model", "usage", "stop_reason")
            if key in result
        }
        return GenerationResponse(raw_text=raw_text, metadata=metadata)


def create_adapter(model_config: dict[str, Any], timeout: float = 90) -> ModelAdapter:
    provider = str(model_config.get("provider", "")).lower()
    model = str(model_config.get("model", "")).strip()
    api_key_env = str(model_config.get("api_key_env", "")).strip()
    if not model or not api_key_env:
        raise ValueError("Each model needs a non-empty 'model' and 'api_key_env'")

    if provider == "openai_compatible":
        return OpenAICompatibleAdapter(
            model=model,
            api_key_env=api_key_env,
            endpoint=str(model_config.get("endpoint", "https://api.openai.com/v1")),
            timeout=timeout,
        )
    if provider == "anthropic":
        return AnthropicAdapter(
            model=model,
            api_key_env=api_key_env,
            endpoint=str(model_config.get("endpoint", "https://api.anthropic.com/v1")),
            api_version=str(model_config.get("api_version", "2023-06-01")),
            timeout=timeout,
        )
    raise ValueError(
        f"Unsupported provider {provider!r}; use 'openai_compatible' or 'anthropic'"
    )
