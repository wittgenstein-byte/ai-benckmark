"""Async concurrent benchmark runner for vLLM workloads."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from llm_benchmark.backends import InferenceBackend, create_backend
from llm_benchmark.config import BenchmarkConfig, ModelProfile
from llm_benchmark.metrics import BenchmarkMetrics, JobTelemetry, build_job_telemetry, compute_benchmark_metrics
from llm_benchmark.monitor import (
    BottleneckDiagnosis,
    SystemMonitor,
    SystemSummary,
    diagnose_bottleneck,
    summarize_system_samples,
)
from llm_benchmark.workload import PromptItem


@dataclass
class BenchmarkRunResult:
    """Complete output from benchmarking one ModelProfile."""

    model: ModelProfile
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    system: SystemSummary
    bottleneck: BottleneckDiagnosis
    jobs: list[JobTelemetry]


async def run_benchmark(
    model: ModelProfile,
    config: BenchmarkConfig,
    workload: list[PromptItem],
    backend: InferenceBackend | None = None,
    verbose: bool = True,
) -> BenchmarkRunResult:
    """Execute a full concurrent benchmark run against a decoupled InferenceBackend."""
    owns_backend = backend is None
    active_backend = backend or create_backend(model, concurrency=config.concurrency)

    try:
        if verbose:
            print(f"\n[Benchmark] Model: {model.name} ({model.model_id}) | Backend: {model.backend_type}")
            print(f"[Benchmark] Endpoint: {model.base_url} ({model.endpoint_type})")
            print(f"[Benchmark] Target: {len(workload)} jobs @ {config.concurrency} concurrent requests")

        healthy = await active_backend.health_check(model)
        if not healthy:
            raise ConnectionError(
                f"vLLM endpoint {model.base_url}/models is unreachable or returned non-200 for '{model.name}'."
            )

        if config.warmup_jobs > 0:
            warmup_count = min(config.warmup_jobs, len(workload))
            if verbose:
                print(f"[Benchmark] Running {warmup_count} warmup job(s)...")
            warmup_sem = asyncio.Semaphore(min(config.concurrency, warmup_count))

            async def _run_warmup(item: PromptItem) -> None:
                async with warmup_sem:
                    await active_backend.generate(item.prompt, model, config.request_timeout_s)

            await asyncio.gather(*[_run_warmup(workload[i]) for i in range(warmup_count)])

        semaphore = asyncio.Semaphore(config.concurrency)
        results: list[JobTelemetry | None] = [None] * len(workload)
        completed = 0
        progress_step = max(1, len(workload) // 10)
        monitor = SystemMonitor(interval_s=config.monitor_interval_s)

        async def _worker(idx: int, item: PromptItem) -> None:
            nonlocal completed
            async with semaphore:
                res = await active_backend.generate(item.prompt, model, config.request_timeout_s)
                results[idx] = build_job_telemetry(item.job_id, res)
                completed += 1
                if verbose and (completed % progress_step == 0 or completed == len(workload)):
                    print(f"  Progress: {completed}/{len(workload)} jobs finished")

        monitor.start()
        wall_start = time.perf_counter()
        try:
            await asyncio.gather(*[_worker(i, item) for i, item in enumerate(workload)])
        finally:
            total_elapsed = time.perf_counter() - wall_start
            samples = monitor.stop()

        final_jobs = [j for j in results if j is not None]
        metrics = compute_benchmark_metrics(final_jobs, total_elapsed)
        system_summary = summarize_system_samples(samples)
        bottleneck = diagnose_bottleneck(system_summary)

        return BenchmarkRunResult(
            model=model,
            config=config,
            metrics=metrics,
            system=system_summary,
            bottleneck=bottleneck,
            jobs=final_jobs,
        )
    finally:
        if owns_backend:
            await active_backend.close()
