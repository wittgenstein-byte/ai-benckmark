"""Decoupled inference backends for vLLM OpenAI-compatible servers and mock testing."""

from __future__ import annotations

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm_benchmark.config import ModelProfile


@dataclass
class GenerationResult:
    """Telemetry from a single inference request."""

    success: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_latency_s: float
    output_text: str = ""
    error: str | None = None


class InferenceBackend(ABC):
    """Abstract interface decoupling the benchmark runner from the serving engine."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: ModelProfile,
        timeout_s: float,
    ) -> GenerationResult:
        """Execute one streaming inference request and return latency/token telemetry."""

    @abstractmethod
    async def health_check(self, model: ModelProfile) -> bool:
        """Verify the backend endpoint is reachable before starting the benchmark."""

    @abstractmethod
    async def close(self) -> None:
        """Release underlying network/client resources."""


def _extract_chunk_text(chunk: dict[str, Any], endpoint_type: str) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if endpoint_type == "completions":
        return str(choice.get("text") or "")
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if content:
        return str(content)
    reasoning = delta.get("reasoning_content")
    if reasoning:
        return str(reasoning)
    return ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


class VLLMOpenAIBackend(InferenceBackend):
    """Async streaming client for vLLM's OpenAI-compatible HTTP API."""

    def __init__(self, max_connections: int = 64) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for VLLMOpenAIBackend. Install it with `pip install httpx`."
            ) from exc

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._httpx = httpx
        self._client = httpx.AsyncClient(limits=limits)

    async def health_check(self, model: ModelProfile) -> bool:
        headers = {"Authorization": f"Bearer {model.api_key}"}
        try:
            resp = await self._client.get(
                f"{model.base_url}/models",
                headers=headers,
                timeout=10.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        model: ModelProfile,
        timeout_s: float,
    ) -> GenerationResult:
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json",
        }

        if model.endpoint_type == "completions":
            url = f"{model.base_url}/completions"
            full_prompt = f"{model.system_prompt}\n\n{prompt}" if model.system_prompt else prompt
            payload: dict[str, Any] = {
                "model": model.model_id,
                "prompt": full_prompt,
                "temperature": model.temperature,
                "max_tokens": model.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        else:
            url = f"{model.base_url}/chat/completions"
            messages: list[dict[str, str]] = []
            if model.system_prompt:
                messages.append({"role": "system", "content": model.system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": model.model_id,
                "messages": messages,
                "temperature": model.temperature,
                "max_tokens": model.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

        if model.extra_body:
            payload.update(model.extra_body)

        start_t = time.perf_counter()
        first_token_t: float | None = None
        output_parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    elapsed = time.perf_counter() - start_t
                    return GenerationResult(
                        success=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        ttft_s=0.0,
                        total_latency_s=elapsed,
                        error=f"HTTP {response.status_code}: {body.decode('utf-8', errors='replace')[:300]}",
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("prompt_tokens") is not None:
                            prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            completion_tokens = int(usage["completion_tokens"])

                    text = _extract_chunk_text(chunk, model.endpoint_type)
                    if text:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        output_parts.append(text)

            end_t = time.perf_counter()
            total_latency = end_t - start_t
            ttft = (first_token_t - start_t) if first_token_t is not None else total_latency
            output_text = "".join(output_parts)

            return GenerationResult(
                success=True,
                prompt_tokens=prompt_tokens if prompt_tokens is not None else _estimate_tokens(prompt),
                completion_tokens=(
                    completion_tokens if completion_tokens is not None else _estimate_tokens(output_text)
                ),
                ttft_s=ttft,
                total_latency_s=total_latency,
                output_text=output_text,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_t
            return GenerationResult(
                success=False,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_s=0.0,
                total_latency_s=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def close(self) -> None:
        await self._client.aclose()


class MockBackend(InferenceBackend):
    """Simulated vLLM backend for dry-runs and unit tests without a GPU."""

    def __init__(
        self,
        prefill_tps: float = 12000.0,
        decode_tps: float = 65.0,
        completion_tokens: int = 128,
        jitter: float = 0.05,
        seed: int = 42,
    ) -> None:
        self._prefill_tps = prefill_tps
        self._decode_tps = decode_tps
        self._completion_tokens = completion_tokens
        self._jitter = jitter
        self._rng = random.Random(seed)

    async def health_check(self, model: ModelProfile) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        model: ModelProfile,
        timeout_s: float,
    ) -> GenerationResult:
        prompt_tokens = _estimate_tokens(prompt)
        out_tokens = min(model.max_tokens, self._completion_tokens)
        jitter_factor = 1.0 + self._rng.uniform(-self._jitter, self._jitter)

        ttft_s = max(0.002, (prompt_tokens / self._prefill_tps) * jitter_factor)
        decode_s = max(0.005, (out_tokens / self._decode_tps) * jitter_factor)

        if ttft_s + decode_s > timeout_s:
            await asyncio.sleep(timeout_s)
            return GenerationResult(
                success=False,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_s=0.0,
                total_latency_s=timeout_s,
                error="TimeoutError: simulated timeout",
            )

        start_t = time.perf_counter()
        await asyncio.sleep(ttft_s)
        actual_ttft = time.perf_counter() - start_t
        await asyncio.sleep(decode_s)
        total_latency = time.perf_counter() - start_t

        return GenerationResult(
            success=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=out_tokens,
            ttft_s=actual_ttft,
            total_latency_s=total_latency,
            output_text=" ".join(["summary"] * min(out_tokens, 16)),
        )

    async def close(self) -> None:
        return None


def create_backend(model: ModelProfile, concurrency: int) -> InferenceBackend:
    """Factory returning the appropriate InferenceBackend for a ModelProfile."""
    if model.backend_type == "mock":
        return MockBackend()
    if model.backend_type in {"vllm", "openai", "openai-compatible"}:
        return VLLMOpenAIBackend(max_connections=max(concurrency + 16, 64))
    raise ValueError(f"Unsupported backend_type: '{model.backend_type}'")
