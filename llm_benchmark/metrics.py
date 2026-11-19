"""Metrics calculation for TTFT, prefill speed, generation throughput, and jobs/day."""

from __future__ import annotations

import math
from dataclasses import dataclass

from llm_benchmark.backends import GenerationResult


@dataclass
class JobTelemetry:
    """Computed metrics for a single benchmark job."""

    job_id: str
    success: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_latency_s: float
    prefill_tps: float
    decode_tps: float
    tpot_ms: float
    error: str | None = None


@dataclass
class LatencyStats:
    """Distribution summary for a numeric metric."""

    min: float
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


@dataclass
class BenchmarkMetrics:
    """Aggregate benchmark metrics across all jobs."""

    total_jobs: int
    succeeded_jobs: int
    failed_jobs: int
    success_rate_pct: float
    total_elapsed_s: float
    jobs_per_second: float
    jobs_per_hour: float
    jobs_per_day: float
    total_prompt_tokens: int
    total_completion_tokens: int
    aggregate_prefill_tps: float
    aggregate_generation_tps: float
    aggregate_total_tps: float
    ttft_ms: LatencyStats
    e2e_latency_s: LatencyStats
    tpot_ms: LatencyStats
    prefill_tps_per_req: LatencyStats
    decode_tps_per_req: LatencyStats


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def summarize_distribution(values: list[float]) -> LatencyStats:
    """Compute min, mean, p50, p95, p99, max for a list of floats."""
    if not values:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = sorted(values)
    return LatencyStats(
        min=s[0],
        mean=sum(s) / len(s),
        p50=_percentile(s, 50.0),
        p95=_percentile(s, 95.0),
        p99=_percentile(s, 99.0),
        max=s[-1],
    )


def build_job_telemetry(job_id: str, res: GenerationResult) -> JobTelemetry:
    """Convert a raw GenerationResult into per-job performance telemetry."""
    if not res.success:
        return JobTelemetry(
            job_id=job_id,
            success=False,
            prompt_tokens=0,
            completion_tokens=0,
            ttft_s=0.0,
            total_latency_s=res.total_latency_s,
            prefill_tps=0.0,
            decode_tps=0.0,
            tpot_ms=0.0,
            error=res.error,
        )

    ttft = max(res.ttft_s, 1e-6)
    prefill_tps = res.prompt_tokens / ttft if res.prompt_tokens > 0 else 0.0

    decode_time = max(res.total_latency_s - res.ttft_s, 1e-6)
    if res.completion_tokens > 1:
        effective_decode_tokens = res.completion_tokens - 1
        decode_tps = effective_decode_tokens / decode_time
        tpot_ms = (decode_time / effective_decode_tokens) * 1000.0
    elif res.completion_tokens == 1:
        decode_tps = 1.0 / max(res.total_latency_s, 1e-6)
        tpot_ms = res.total_latency_s * 1000.0
    else:
        decode_tps = 0.0
        tpot_ms = 0.0

    return JobTelemetry(
        job_id=job_id,
        success=True,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        ttft_s=res.ttft_s,
        total_latency_s=res.total_latency_s,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        tpot_ms=tpot_ms,
        error=None,
    )


def compute_benchmark_metrics(
    jobs: list[JobTelemetry],
    total_elapsed_s: float,
) -> BenchmarkMetrics:
    """Aggregate all per-job telemetry into workload capacity and throughput metrics."""
    elapsed = max(total_elapsed_s, 1e-6)
    total_jobs = len(jobs)
    succeeded = [j for j in jobs if j.success]
    failed_count = total_jobs - len(succeeded)
    success_rate = (len(succeeded) / total_jobs * 100.0) if total_jobs > 0 else 0.0

    jobs_per_sec = len(succeeded) / elapsed
    jobs_per_hr = jobs_per_sec * 3600.0
    jobs_per_day = jobs_per_sec * 86400.0

    total_prompt_tokens = sum(j.prompt_tokens for j in succeeded)
    total_completion_tokens = sum(j.completion_tokens for j in succeeded)

    return BenchmarkMetrics(
        total_jobs=total_jobs,
        succeeded_jobs=len(succeeded),
        failed_jobs=failed_count,
        success_rate_pct=success_rate,
        total_elapsed_s=elapsed,
        jobs_per_second=jobs_per_sec,
        jobs_per_hour=jobs_per_hr,
        jobs_per_day=jobs_per_day,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        aggregate_prefill_tps=total_prompt_tokens / elapsed,
        aggregate_generation_tps=total_completion_tokens / elapsed,
        aggregate_total_tps=(total_prompt_tokens + total_completion_tokens) / elapsed,
        ttft_ms=summarize_distribution([j.ttft_s * 1000.0 for j in succeeded]),
        e2e_latency_s=summarize_distribution([j.total_latency_s for j in succeeded]),
        tpot_ms=summarize_distribution([j.tpot_ms for j in succeeded if j.completion_tokens > 0]),
        prefill_tps_per_req=summarize_distribution([j.prefill_tps for j in succeeded]),
        decode_tps_per_req=summarize_distribution([j.decode_tps for j in succeeded if j.completion_tokens > 0]),
    )
