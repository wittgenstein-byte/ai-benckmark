"""Unit and integration tests for the decoupled vLLM benchmark suite."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_benchmark.backends import GenerationResult, MockBackend
from llm_benchmark.config import BenchmarkConfig, ModelProfile, load_model_profiles
from llm_benchmark.metrics import build_job_telemetry, compute_benchmark_metrics
from llm_benchmark.monitor import SystemSummary, diagnose_bottleneck
from llm_benchmark.report import save_reports
from llm_benchmark.runner import run_benchmark
from llm_benchmark.workload import load_workload


class TestConfigAndWorkload(unittest.TestCase):
    def test_load_json_profiles_and_env_expansion(self) -> None:
        os.environ["TEST_VLLM_KEY"] = "secret-123"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg_path = Path(tmp) / "models.json"
                cfg_path.write_text(
                    json.dumps({
                        "models": [
                            {
                                "name": "qwen-14b",
                                "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                                "base_url": "http://localhost:8000/v1/",
                                "api_key": "${TEST_VLLM_KEY}",
                                "max_tokens": 256,
                            }
                        ]
                    }),
                    encoding="utf-8",
                )
                profiles = load_model_profiles(cfg_path)
                self.assertIn("qwen-14b", profiles)
                p = profiles["qwen-14b"]
                self.assertEqual(p.base_url, "http://localhost:8000/v1")
                self.assertEqual(p.api_key, "secret-123")
                self.assertEqual(p.max_tokens, 256)
        finally:
            os.environ.pop("TEST_VLLM_KEY", None)

    def test_load_workload_from_synthea_jsonl_cycles_to_total_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "patients_summary.jsonl"
            jsonl_path.write_text(
                "\n".join([
                    json.dumps({"patient_id": "p-1", "clinical_summary_md": "Patient 1 clinical note."}),
                    json.dumps({"patient_id": "p-2", "clinical_summary_md": "Patient 2 clinical note."}),
                ])
                + "\n",
                encoding="utf-8",
            )
            items = load_workload(str(jsonl_path), total_jobs=5, unique_prompts=True)
            self.assertEqual(len(items), 5)
            self.assertIn("Patient 1 clinical note.", items[0].prompt)
            self.assertIn("Patient 2 clinical note.", items[1].prompt)
            self.assertIn("Benchmark Request #00003", items[2].prompt)


class TestMetricsAndBottleneck(unittest.TestCase):
    def test_compute_metrics_jobs_per_day_and_throughput(self) -> None:
        jobs = [
            build_job_telemetry(
                f"j-{i}",
                GenerationResult(
                    success=True,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    ttft_s=0.1,
                    total_latency_s=1.1,
                ),
            )
            for i in range(10)
        ]
        metrics = compute_benchmark_metrics(jobs, total_elapsed_s=2.0)
        self.assertEqual(metrics.succeeded_jobs, 10)
        self.assertAlmostEqual(metrics.jobs_per_second, 5.0, places=3)
        self.assertAlmostEqual(metrics.jobs_per_day, 5.0 * 86400.0, places=1)
        self.assertAlmostEqual(metrics.aggregate_prefill_tps, 5000.0, places=1)
        self.assertAlmostEqual(metrics.aggregate_generation_tps, 500.0, places=1)
        self.assertAlmostEqual(metrics.ttft_ms.p50, 100.0, places=1)

    def test_diagnose_bottleneck_classifications(self) -> None:
        gpu_bound = SystemSummary(
            sample_count=10,
            cpu_avg_pct=25.0,
            cpu_peak_pct=40.0,
            ram_avg_pct=30.0,
            ram_peak_pct=32.0,
            ram_peak_used_gb=20.0,
            ram_total_gb=64.0,
            gpu_detected=True,
            gpu_name="NVIDIA RTX 6000 Ada Generation",
            gpu_count=1,
            gpu_util_avg_pct=94.0,
            gpu_util_peak_pct=99.0,
            gpu_mem_util_avg_pct=70.0,
            vram_peak_used_mb=38000.0,
            vram_total_mb=49140.0,
            vram_peak_pct=77.3,
            gpu_temp_peak_c=74.0,
            gpu_power_avg_w=275.0,
        )
        diag_gpu = diagnose_bottleneck(gpu_bound)
        self.assertIn("GPU Compute Bound", diag_gpu.primary_bottleneck)

        cpu_bound = SystemSummary(
            sample_count=10,
            cpu_avg_pct=92.0,
            cpu_peak_pct=98.0,
            ram_avg_pct=40.0,
            ram_peak_pct=45.0,
            ram_peak_used_gb=28.0,
            ram_total_gb=64.0,
            gpu_detected=True,
            gpu_name="NVIDIA RTX 6000 Ada Generation",
            gpu_count=1,
            gpu_util_avg_pct=42.0,
            gpu_util_peak_pct=55.0,
            gpu_mem_util_avg_pct=30.0,
            vram_peak_used_mb=24000.0,
            vram_total_mb=49140.0,
            vram_peak_pct=48.8,
            gpu_temp_peak_c=60.0,
            gpu_power_avg_w=140.0,
        )
        diag_cpu = diagnose_bottleneck(cpu_bound)
        self.assertIn("CPU", diag_cpu.primary_bottleneck)


class TestEndToEndMockBenchmark(unittest.TestCase):
    def test_run_benchmark_and_save_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ModelProfile(
                name="mock-rtx6000",
                model_id="mock-model",
                backend_type="mock",
                max_tokens=64,
            )
            config = BenchmarkConfig(
                total_jobs=30,
                concurrency=10,
                warmup_jobs=2,
                monitor_interval_s=0.1,
                output_dir=tmp,
                synthetic_prompt_tokens=400,
            )
            workload = load_workload("synthetic", total_jobs=config.total_jobs, synthetic_tokens=400)
            backend = MockBackend(prefill_tps=40000.0, decode_tps=800.0, completion_tokens=32, jitter=0.0)

            result = asyncio.run(
                run_benchmark(model=model, config=config, workload=workload, backend=backend, verbose=False)
            )
            self.assertEqual(result.metrics.succeeded_jobs, 30)
            self.assertEqual(result.metrics.failed_jobs, 0)
            self.assertGreater(result.metrics.jobs_per_day, 0.0)

            written = save_reports([result], tmp)
            self.assertTrue(written["json"].exists())
            self.assertTrue(written["markdown"].exists())
            self.assertTrue(written["csv_mock-rtx6000"].exists())


if __name__ == "__main__":
    unittest.main()
