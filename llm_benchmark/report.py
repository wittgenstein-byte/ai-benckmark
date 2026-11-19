"""Console, JSON, CSV, and Markdown reporting for vLLM benchmark runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from llm_benchmark.runner import BenchmarkRunResult


def print_run_summary(run: BenchmarkRunResult) -> None:
    """Print a concise summary of one benchmark run to stdout."""
    m = run.metrics
    s = run.system
    b = run.bottleneck

    print("\n" + "=" * 72)
    print(f"BENCHMARK SUMMARY: {run.model.name} ({run.model.model_id})")
    print("=" * 72)
    print(
        f"Workload       : {m.succeeded_jobs}/{m.total_jobs} succeeded "
        f"({m.success_rate_pct:.1f}%) | Concurrency: {run.config.concurrency}"
    )
    print(f"Total Elapsed  : {m.total_elapsed_s:.2f} s")
    print(
        f"Capacity       : {m.jobs_per_second:.2f} jobs/s | "
        f"{m.jobs_per_hour:,.0f} jobs/hr | {m.jobs_per_day:,.0f} jobs/day"
    )
    print("-" * 72)
    print(
        f"Prefill (TTFT) : p50={m.ttft_ms.p50:.1f} ms | p95={m.ttft_ms.p95:.1f} ms | "
        f"p99={m.ttft_ms.p99:.1f} ms | mean={m.ttft_ms.mean:.1f} ms"
    )
    print(
        f"Prefill Speed  : {m.aggregate_prefill_tps:,.1f} tok/s (aggregate) | "
        f"{m.prefill_tps_per_req.p50:,.1f} tok/s (p50/req)"
    )
    print(
        f"Generation TPS : {m.aggregate_generation_tps:,.1f} output tok/s (aggregate @ {run.config.concurrency} users) | "
        f"{m.decode_tps_per_req.p50:.1f} tok/s (p50/req)"
    )
    print(
        f"TPOT / ITL     : p50={m.tpot_ms.p50:.2f} ms/tok | p95={m.tpot_ms.p95:.2f} ms/tok"
    )
    print(
        f"E2E Latency    : p50={m.e2e_latency_s.p50:.2f} s | p95={m.e2e_latency_s.p95:.2f} s | "
        f"max={m.e2e_latency_s.max:.2f} s"
    )
    print("-" * 72)
    print(
        f"Host CPU / RAM : CPU avg={s.cpu_avg_pct:.1f}% (peak {s.cpu_peak_pct:.1f}%) | "
        f"RAM peak={s.ram_peak_used_gb:.1f}/{s.ram_total_gb:.1f} GB ({s.ram_peak_pct:.1f}%)"
    )
    if s.gpu_detected:
        print(
            f"GPU ({s.gpu_name} x{s.gpu_count}): "
            f"Compute avg={s.gpu_util_avg_pct:.1f}% (peak {s.gpu_util_peak_pct:.1f}%) | "
            f"VRAM peak={s.vram_peak_used_mb:,.0f}/{s.vram_total_mb:,.0f} MB ({s.vram_peak_pct:.1f}%)"
        )
    else:
        print("GPU Telemetry  : Not detected locally (nvidia-smi unavailable)")
    print(f"Bottleneck     : {b.primary_bottleneck} [{b.severity.upper()}]")
    print(f"  Detail       : {b.explanation}")
    print("=" * 72)


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name)


def save_reports(runs: list[BenchmarkRunResult], output_dir: str | Path) -> dict[str, Path]:
    """Write JSON, per-job CSVs, and Markdown comparison report to `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    json_path = out / "benchmark_results.json"
    json_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "model": asdict(r.model),
                "config": asdict(r.config),
                "metrics": asdict(r.metrics),
                "system": asdict(r.system),
                "bottleneck": asdict(r.bottleneck),
            }
            for r in runs
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    written["json"] = json_path

    for r in runs:
        safe_name = _sanitize_filename(r.model.name)
        csv_path = out / f"benchmark_jobs_{safe_name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "job_id",
                "success",
                "prompt_tokens",
                "completion_tokens",
                "ttft_ms",
                "total_latency_s",
                "prefill_tps",
                "decode_tps",
                "tpot_ms",
                "error",
            ])
            for j in r.jobs:
                writer.writerow([
                    j.job_id,
                    j.success,
                    j.prompt_tokens,
                    j.completion_tokens,
                    f"{j.ttft_s * 1000.0:.2f}",
                    f"{j.total_latency_s:.4f}",
                    f"{j.prefill_tps:.2f}",
                    f"{j.decode_tps:.2f}",
                    f"{j.tpot_ms:.2f}",
                    j.error or "",
                ])
        written[f"csv_{safe_name}"] = csv_path

    md_path = out / "benchmark_report.md"
    md_lines = [
        "# Local AI / vLLM Benchmark Report",
        "",
        f"Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "## Model Comparison Summary",
        "",
        "| Model | Jobs (OK/Total) | Concurrency | Elapsed (s) | Jobs / Day | TTFT p50 (ms) | TTFT p95 (ms) | Prefill (tok/s) | Gen Throughput (tok/s) | CPU Avg (%) | RAM Peak (%) | GPU Avg (%) | VRAM Peak (%) | Bottleneck |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        m = r.metrics
        s = r.system
        gpu_str = f"{s.gpu_util_avg_pct:.1f}%" if s.gpu_detected else "N/A"
        vram_str = f"{s.vram_peak_pct:.1f}%" if s.gpu_detected else "N/A"
        md_lines.append(
            f"| `{r.model.name}` | {m.succeeded_jobs}/{m.total_jobs} | {r.config.concurrency} | "
            f"{m.total_elapsed_s:.2f} | **{m.jobs_per_day:,.0f}** | {m.ttft_ms.p50:.1f} | "
            f"{m.ttft_ms.p95:.1f} | {m.aggregate_prefill_tps:,.1f} | **{m.aggregate_generation_tps:,.1f}** | "
            f"{s.cpu_avg_pct:.1f}% | {s.ram_peak_pct:.1f}% | {gpu_str} | {vram_str} | {r.bottleneck.primary_bottleneck} |"
        )

    for r in runs:
        m = r.metrics
        s = r.system
        b = r.bottleneck
        md_lines.extend([
            "",
            f"## Detailed Results: `{r.model.name}`",
            "",
            f"- **Model ID**: `{r.model.model_id}` (`{r.model.base_url}`)",
            f"- **Target Workload**: {m.succeeded_jobs}/{m.total_jobs} jobs @ {r.config.concurrency} concurrent requests in **{m.total_elapsed_s:.2f} s**",
            f"- **Daily Capacity**: **{m.jobs_per_day:,.0f} jobs/day** ({m.jobs_per_hour:,.0f} jobs/hr, {m.jobs_per_second:.2f} jobs/s)",
            f"- **Prompt Processing (TTFT)**: p50 `{m.ttft_ms.p50:.1f} ms`, p95 `{m.ttft_ms.p95:.1f} ms`, p99 `{m.ttft_ms.p99:.1f} ms` | Aggregate Prefill: `{m.aggregate_prefill_tps:,.1f} tok/s`",
            f"- **Generation Throughput**: **{m.aggregate_generation_tps:,.1f} output tok/s** (system-wide) | Per-request p50 `{m.decode_tps_per_req.p50:.1f} tok/s` | TPOT p50 `{m.tpot_ms.p50:.2f} ms/tok`",
            f"- **Host Resources**: CPU avg `{s.cpu_avg_pct:.1f}%` (peak `{s.cpu_peak_pct:.1f}%`), RAM peak `{s.ram_peak_used_gb:.1f}/{s.ram_total_gb:.1f} GB` (`{s.ram_peak_pct:.1f}%`)",
        ])
        if s.gpu_detected:
            md_lines.append(
                f"- **GPU Resources (`{s.gpu_name}` x{s.gpu_count})**: Compute avg `{s.gpu_util_avg_pct:.1f}%` (peak `{s.gpu_util_peak_pct:.1f}%`), "
                f"VRAM peak `{s.vram_peak_used_mb:,.0f}/{s.vram_total_mb:,.0f} MB` (`{s.vram_peak_pct:.1f}%`)"
            )
        md_lines.extend([
            f"- **Bottleneck Diagnosis**: **{b.primary_bottleneck}** — {b.explanation}",
        ])
        for rec in b.recommendations:
            md_lines.append(f"  - Recommendation: {rec}")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written["markdown"] = md_path
    return written
