"""CLI entrypoint for running decoupled vLLM benchmarks (1,000 jobs @ 30 concurrency)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from llm_benchmark.config import BenchmarkConfig, ModelProfile, load_model_profiles
from llm_benchmark.report import print_run_summary, save_reports
from llm_benchmark.runner import BenchmarkRunResult, run_benchmark
from llm_benchmark.workload import load_workload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local vLLM models (RTX 6000 series): Jobs/Day, TTFT, Decode TPS, and System Bottleneck.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model profiles JSON or YAML file (e.g. configs/models.example.json).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model profile name(s) from --config to benchmark sequentially.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all non-mock model profiles defined in --config.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Direct vLLM model ID (used when --config is not provided, or to override).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="vLLM OpenAI-compatible base URL (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for vLLM server (default: EMPTY).",
    )
    parser.add_argument(
        "--endpoint-type",
        choices=["chat", "completions"],
        default=None,
        help="OpenAI endpoint type: 'chat' (/chat/completions) or 'completions' (/completions).",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "mock"],
        default=None,
        help="Inference backend override ('vllm' or 'mock' for dry-run).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max output tokens per job.",
    )
    parser.add_argument(
        "--total-jobs",
        type=int,
        default=1000,
        help="Total number of jobs to execute (default: 1000).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="Number of concurrent requests (default: 30).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup requests before timing starts (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="System CPU/RAM/GPU sampling interval in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="synthetic",
        help="Path to JSONL (e.g. etl_output/patients_summary.jsonl) or 'synthetic' (default: synthetic).",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=1500,
        help="Target estimated tokens per prompt when --prompt-source is 'synthetic' (default: 1500).",
    )
    parser.add_argument(
        "--no-unique-prompts",
        action="store_true",
        help="Disable per-request unique ID headers (enables vLLM prefix caching across repeated prompts).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_output",
        help="Directory to save JSON, CSV, and Markdown reports (default: ./benchmark_output).",
    )
    return parser.parse_args(argv)


def _resolve_models(args: argparse.Namespace) -> list[ModelProfile]:
    selected: list[ModelProfile] = []

    if args.config:
        profiles = load_model_profiles(args.config)
        if args.all_models:
            selected = [p for p in profiles.values() if p.backend_type != "mock"]
            if not selected:
                selected = list(profiles.values())
        elif args.models:
            for name in args.models:
                if name not in profiles:
                    available = ", ".join(profiles.keys())
                    raise KeyError(f"Model '{name}' not found in {args.config}. Available: {available}")
                selected.append(profiles[name])
        else:
            first_key = next(iter(profiles))
            selected = [profiles[first_key]]
    else:
        model_id = args.model_id or "default-vllm-model"
        selected = [
            ModelProfile(
                name=model_id.split("/")[-1],
                model_id=model_id,
                base_url=(args.base_url or "http://localhost:8000/v1").rstrip("/"),
                api_key=args.api_key or "EMPTY",
                endpoint_type=args.endpoint_type or "chat",
                backend_type=args.backend or "vllm",
                max_tokens=args.max_tokens or 512,
            )
        ]

    # Apply CLI overrides uniformly across selected profiles
    for p in selected:
        if args.model_id and args.config:
            p.model_id = args.model_id
        if args.base_url:
            p.base_url = args.base_url.rstrip("/")
        if args.api_key:
            p.api_key = args.api_key
        if args.endpoint_type:
            p.endpoint_type = args.endpoint_type
        if args.backend:
            p.backend_type = args.backend
        if args.max_tokens is not None:
            p.max_tokens = args.max_tokens

    return selected


async def _async_main(args: argparse.Namespace) -> int:
    models = _resolve_models(args)
    config = BenchmarkConfig(
        total_jobs=args.total_jobs,
        concurrency=args.concurrency,
        warmup_jobs=args.warmup,
        request_timeout_s=args.timeout,
        monitor_interval_s=args.monitor_interval,
        output_dir=args.output_dir,
        prompt_source=args.prompt_source,
        synthetic_prompt_tokens=args.prompt_tokens,
        unique_prompts=not args.no_unique_prompts,
    )

    workload = load_workload(
        source=config.prompt_source,
        total_jobs=config.total_jobs,
        synthetic_tokens=config.synthetic_prompt_tokens,
        unique_prompts=config.unique_prompts,
    )
    print(
        f"[Workload] Prepared {len(workload)} jobs from '{config.prompt_source}' "
        f"(avg ~{sum(w.estimated_tokens for w in workload) // len(workload)} est. input tokens/job)"
    )

    runs: list[BenchmarkRunResult] = []
    for model in models:
        run_result = await run_benchmark(model=model, config=config, workload=workload, verbose=True)
        print_run_summary(run_result)
        runs.append(run_result)

    written = save_reports(runs, Path(config.output_dir))
    print("\nSaved reports:")
    for label, path in written.items():
        print(f"  - {label}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
