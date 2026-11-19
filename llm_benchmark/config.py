"""Configuration models and decoupled profile loaders for vLLM benchmarking."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical AI assistant. Summarize the patient's longitudinal "
    "medical record concisely, highlighting active conditions, key medications, "
    "recent abnormal labs/vitals, and recommended follow-up."
)


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} placeholders in string config values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


@dataclass
class ModelProfile:
    """Decoupled model & endpoint configuration for vLLM or compatible servers."""

    name: str
    model_id: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    endpoint_type: str = "chat"  # "chat" (/chat/completions) or "completions" (/completions)
    backend_type: str = "vllm"  # "vllm" or "mock"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.0
    max_tokens: int = 512
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        expanded = _expand_env_vars(data)
        return cls(
            name=str(expanded["name"]),
            model_id=str(expanded.get("model_id", expanded["name"])),
            base_url=str(expanded.get("base_url", "http://localhost:8000/v1")).rstrip("/"),
            api_key=str(expanded.get("api_key", "EMPTY")) or "EMPTY",
            endpoint_type=str(expanded.get("endpoint_type", "chat")).lower(),
            backend_type=str(expanded.get("backend_type", "vllm")).lower(),
            system_prompt=str(expanded.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            temperature=float(expanded.get("temperature", 0.0)),
            max_tokens=int(expanded.get("max_tokens", 512)),
            extra_body=dict(expanded.get("extra_body", {})),
        )


@dataclass
class BenchmarkConfig:
    """Runtime parameters for a benchmark run."""

    total_jobs: int = 1000
    concurrency: int = 30
    warmup_jobs: int = 5
    request_timeout_s: float = 300.0
    monitor_interval_s: float = 0.5
    output_dir: str = "./benchmark_output"
    prompt_source: str = "synthetic"
    synthetic_prompt_tokens: int = 1500
    unique_prompts: bool = True


def load_model_profiles(config_path: str | Path) -> dict[str, ModelProfile]:
    """Load model profiles from a JSON or YAML file keyed by profile name."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load .yaml/.yml configs. "
                "Install it with `pip install pyyaml` or use a .json config."
            ) from exc
        payload = yaml.safe_load(raw_text)
    else:
        raise ValueError(f"Unsupported config format '{suffix}'. Use .json, .yaml, or .yml.")

    raw_models = payload.get("models", payload) if isinstance(payload, dict) else payload
    profiles: dict[str, ModelProfile] = {}

    if isinstance(raw_models, list):
        for item in raw_models:
            profile = ModelProfile.from_dict(item)
            profiles[profile.name] = profile
    elif isinstance(raw_models, dict):
        for key, item in raw_models.items():
            merged = dict(item)
            merged.setdefault("name", key)
            profile = ModelProfile.from_dict(merged)
            profiles[profile.name] = profile
    else:
        raise ValueError("Config must contain a 'models' list or mapping.")

    return profiles
