"""Background system resource monitor (CPU, RAM, and NVIDIA GPU via nvidia-smi)."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class GpuSample:
    """Single-instant telemetry for one NVIDIA GPU."""

    index: int
    name: str
    gpu_util_pct: float
    mem_util_pct: float
    vram_used_mb: float
    vram_total_mb: float
    temp_c: float | None = None
    power_w: float | None = None


@dataclass
class SystemSample:
    """Single-instant host + GPU telemetry."""

    timestamp_s: float
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    ram_pct: float
    gpus: list[GpuSample] = field(default_factory=list)


@dataclass
class SystemSummary:
    """Aggregated host and GPU utilization over the benchmark window."""

    sample_count: int
    cpu_avg_pct: float
    cpu_peak_pct: float
    ram_avg_pct: float
    ram_peak_pct: float
    ram_peak_used_gb: float
    ram_total_gb: float
    gpu_detected: bool
    gpu_name: str | None
    gpu_count: int
    gpu_util_avg_pct: float
    gpu_util_peak_pct: float
    gpu_mem_util_avg_pct: float
    vram_peak_used_mb: float
    vram_total_mb: float
    vram_peak_pct: float
    gpu_temp_peak_c: float | None
    gpu_power_avg_w: float | None


@dataclass
class BottleneckDiagnosis:
    """Automated diagnosis of whether the workload is GPU, VRAM, CPU, RAM, or queue bound."""

    primary_bottleneck: str
    severity: str
    explanation: str
    recommendations: list[str]


def _parse_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _query_nvidia_smi(nvidia_smi_bin: str) -> list[GpuSample]:
    cmd = [
        nvidia_smi_bin,
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    gpus: list[GpuSample] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx = int(_parse_float(parts[0]) or 0)
        name = parts[1]
        gpu_util = _parse_float(parts[2]) or 0.0
        mem_util = _parse_float(parts[3]) or 0.0
        vram_used = _parse_float(parts[4]) or 0.0
        vram_total = _parse_float(parts[5]) or 0.0
        temp_c = _parse_float(parts[6]) if len(parts) > 6 else None
        power_w = _parse_float(parts[7]) if len(parts) > 7 else None
        gpus.append(
            GpuSample(
                index=idx,
                name=name,
                gpu_util_pct=gpu_util,
                mem_util_pct=mem_util,
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
                temp_c=temp_c,
                power_w=power_w,
            )
        )
    return gpus


class _LinuxProcCpuReader:
    """Fallback CPU/RAM reader using /proc when psutil is unavailable on Linux."""

    def __init__(self) -> None:
        self._prev_idle = 0
        self._prev_total = 0
        self.read_cpu_pct()

    def read_cpu_pct(self) -> float:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as fh:
                first_line = fh.readline()
            parts = [int(x) for x in first_line.split()[1:]]
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            total = sum(parts)
            diff_idle = idle - self._prev_idle
            diff_total = total - self._prev_total
            self._prev_idle = idle
            self._prev_total = total
            if diff_total <= 0:
                return 0.0
            return max(0.0, min(100.0, 100.0 * (1.0 - diff_idle / diff_total)))
        except Exception:
            return 0.0

    @staticmethod
    def read_ram() -> tuple[float, float, float]:
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    key, val = line.split(":", 1)
                    info[key] = int(val.strip().split()[0])
            total_kb = info.get("MemTotal", 0)
            avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
            used_kb = max(0, total_kb - avail_kb)
            total_gb = total_kb / (1024 * 1024)
            used_gb = used_kb / (1024 * 1024)
            pct = (used_kb / total_kb * 100.0) if total_kb > 0 else 0.0
            return used_gb, total_gb, pct
        except Exception:
            return 0.0, 0.0, 0.0


class SystemMonitor:
    """Daemon thread sampling CPU, RAM, and NVIDIA GPU metrics at a fixed interval."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval_s = max(0.1, interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[SystemSample] = []
        self._lock = threading.Lock()
        self._nvidia_smi = shutil.which("nvidia-smi")

        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._psutil.cpu_percent(interval=None)
            self._proc_reader = None
        except ImportError:
            self._psutil = None
            self._proc_reader = _LinuxProcCpuReader() if os.path.exists("/proc/stat") else None

    def _collect_once(self) -> SystemSample:
        if self._psutil is not None:
            cpu_pct = float(self._psutil.cpu_percent(interval=None))
            vm = self._psutil.virtual_memory()
            ram_used_gb = vm.used / (1024**3)
            ram_total_gb = vm.total / (1024**3)
            ram_pct = float(vm.percent)
        elif self._proc_reader is not None:
            cpu_pct = self._proc_reader.read_cpu_pct()
            ram_used_gb, ram_total_gb, ram_pct = self._proc_reader.read_ram()
        else:
            cpu_pct, ram_used_gb, ram_total_gb, ram_pct = 0.0, 0.0, 0.0, 0.0

        gpus = _query_nvidia_smi(self._nvidia_smi) if self._nvidia_smi else []
        return SystemSample(
            timestamp_s=time.perf_counter(),
            cpu_pct=cpu_pct,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            ram_pct=ram_pct,
            gpus=gpus,
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._collect_once()
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(self._interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="system-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[SystemSample]:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            return list(self._samples)


def summarize_system_samples(samples: list[SystemSample]) -> SystemSummary:
    """Aggregate collected host and GPU samples."""
    if not samples:
        return SystemSummary(
            sample_count=0,
            cpu_avg_pct=0.0,
            cpu_peak_pct=0.0,
            ram_avg_pct=0.0,
            ram_peak_pct=0.0,
            ram_peak_used_gb=0.0,
            ram_total_gb=0.0,
            gpu_detected=False,
            gpu_name=None,
            gpu_count=0,
            gpu_util_avg_pct=0.0,
            gpu_util_peak_pct=0.0,
            gpu_mem_util_avg_pct=0.0,
            vram_peak_used_mb=0.0,
            vram_total_mb=0.0,
            vram_peak_pct=0.0,
            gpu_temp_peak_c=None,
            gpu_power_avg_w=None,
        )

    cpus = [s.cpu_pct for s in samples]
    rams_pct = [s.ram_pct for s in samples]
    rams_gb = [s.ram_used_gb for s in samples]
    ram_total_gb = max((s.ram_total_gb for s in samples), default=0.0)

    gpu_samples = [g for s in samples for g in s.gpus]
    if not gpu_samples:
        return SystemSummary(
            sample_count=len(samples),
            cpu_avg_pct=sum(cpus) / len(cpus),
            cpu_peak_pct=max(cpus),
            ram_avg_pct=sum(rams_pct) / len(rams_pct),
            ram_peak_pct=max(rams_pct),
            ram_peak_used_gb=max(rams_gb),
            ram_total_gb=ram_total_gb,
            gpu_detected=False,
            gpu_name=None,
            gpu_count=0,
            gpu_util_avg_pct=0.0,
            gpu_util_peak_pct=0.0,
            gpu_mem_util_avg_pct=0.0,
            vram_peak_used_mb=0.0,
            vram_total_mb=0.0,
            vram_peak_pct=0.0,
            gpu_temp_peak_c=None,
            gpu_power_avg_w=None,
        )

    gpu_indices = {g.index for g in gpu_samples}
    gpu_name = gpu_samples[0].name
    gpu_utils = [g.gpu_util_pct for g in gpu_samples]
    mem_utils = [g.mem_util_pct for g in gpu_samples]
    vram_used_peak = max(g.vram_used_mb for g in gpu_samples)
    vram_total = max(g.vram_total_mb for g in gpu_samples)
    vram_peak_pct = (vram_used_peak / vram_total * 100.0) if vram_total > 0 else 0.0
    temps = [g.temp_c for g in gpu_samples if g.temp_c is not None]
    powers = [g.power_w for g in gpu_samples if g.power_w is not None]

    return SystemSummary(
        sample_count=len(samples),
        cpu_avg_pct=sum(cpus) / len(cpus),
        cpu_peak_pct=max(cpus),
        ram_avg_pct=sum(rams_pct) / len(rams_pct),
        ram_peak_pct=max(rams_pct),
        ram_peak_used_gb=max(rams_gb),
        ram_total_gb=ram_total_gb,
        gpu_detected=True,
        gpu_name=gpu_name,
        gpu_count=len(gpu_indices),
        gpu_util_avg_pct=sum(gpu_utils) / len(gpu_utils),
        gpu_util_peak_pct=max(gpu_utils),
        gpu_mem_util_avg_pct=sum(mem_utils) / len(mem_utils),
        vram_peak_used_mb=vram_used_peak,
        vram_total_mb=vram_total,
        vram_peak_pct=vram_peak_pct,
        gpu_temp_peak_c=max(temps) if temps else None,
        gpu_power_avg_w=(sum(powers) / len(powers)) if powers else None,
    )


def diagnose_bottleneck(summary: SystemSummary) -> BottleneckDiagnosis:
    """Classify the primary hardware/system bottleneck from observed telemetry."""
    if not summary.gpu_detected:
        if summary.cpu_avg_pct >= 80.0 or summary.cpu_peak_pct >= 90.0:
            return BottleneckDiagnosis(
                primary_bottleneck="CPU Bound (No GPU telemetry detected)",
                severity="high",
                explanation=(
                    f"CPU averaged {summary.cpu_avg_pct:.1f}% (peak {summary.cpu_peak_pct:.1f}%). "
                    "No NVIDIA GPU was detected by nvidia-smi on this host."
                ),
                recommendations=[
                    "Verify benchmark is running on the RTX 6000 host or check nvidia-smi availability.",
                    "If the client is CPU-bound on tokenization/JSON parsing, increase vLLM --max-num-seqs or offload client.",
                ],
            )
        return BottleneckDiagnosis(
            primary_bottleneck="Unknown / Remote GPU (No local nvidia-smi)",
            severity="info",
            explanation=(
                f"Host CPU avg {summary.cpu_avg_pct:.1f}%, RAM peak {summary.ram_peak_pct:.1f}%. "
                "nvidia-smi was not found on this machine (expected if benchmarking a remote vLLM server)."
            ),
            recommendations=[
                "Run the benchmark script on the RTX 6000 machine or inspect GPU metrics on the vLLM host.",
            ],
        )

    if summary.vram_peak_pct >= 92.0 and summary.gpu_util_avg_pct < 80.0:
        return BottleneckDiagnosis(
            primary_bottleneck="GPU VRAM / KV-Cache Bound",
            severity="high",
            explanation=(
                f"Peak VRAM reached {summary.vram_peak_pct:.1f}% "
                f"({summary.vram_peak_used_mb:.0f}/{summary.vram_total_mb:.0f} MB) while GPU compute "
                f"averaged {summary.gpu_util_avg_pct:.1f}%. KV-cache pressure or request preemption is likely limiting concurrency."
            ),
            recommendations=[
                "Use a quantized model (AWQ / FP8) or reduce --max-model-len in vLLM.",
                "Tune --gpu-memory-utilization (e.g. 0.92) or enable FP8 KV cache (--kv-cache-dtype fp8).",
            ],
        )

    if summary.cpu_avg_pct >= 80.0 and summary.gpu_util_avg_pct < 75.0:
        return BottleneckDiagnosis(
            primary_bottleneck="CPU / Queue Scheduling Bound",
            severity="high",
            explanation=(
                f"Host CPU averaged {summary.cpu_avg_pct:.1f}% (peak {summary.cpu_peak_pct:.1f}%) "
                f"while GPU compute averaged only {summary.gpu_util_avg_pct:.1f}%. "
                "Tokenization, HTTP/SSE serialization, or vLLM API server worker threads are bottlenecking GPU feed."
            ),
            recommendations=[
                "Ensure vLLM runs with --disable-log-requests and sufficient CPU cores.",
                "Check if prompt preprocessing/tokenization is saturating a single Python process.",
            ],
        )

    if summary.ram_peak_pct >= 88.0:
        return BottleneckDiagnosis(
            primary_bottleneck="System RAM Bound",
            severity="high",
            explanation=(
                f"System RAM peaked at {summary.ram_peak_pct:.1f}% "
                f"({summary.ram_peak_used_gb:.1f}/{summary.ram_total_gb:.1f} GB)."
            ),
            recommendations=[
                "Check for CPU swap or KV-cache CPU offloading (--swap-space) causing latency spikes.",
            ],
        )

    if summary.gpu_util_avg_pct >= 85.0:
        return BottleneckDiagnosis(
            primary_bottleneck="GPU Compute Bound (Healthy Saturation)",
            severity="normal",
            explanation=(
                f"GPU compute averaged {summary.gpu_util_avg_pct:.1f}% (peak {summary.gpu_util_peak_pct:.1f}%), "
                f"VRAM peaked at {summary.vram_peak_pct:.1f}%, while CPU stayed at {summary.cpu_avg_pct:.1f}%. "
                "The RTX 6000 is efficiently saturated without host CPU/RAM bottlenecks."
            ),
            recommendations=[
                "To increase jobs/day further, test FP8/AWQ quantization, tensor parallelism, or prefix caching.",
            ],
        )

    return BottleneckDiagnosis(
        primary_bottleneck="Queue / Client Concurrency / Network Bound",
        severity="medium",
        explanation=(
            f"GPU compute averaged {summary.gpu_util_avg_pct:.1f}% and CPU averaged {summary.cpu_avg_pct:.1f}% "
            f"(VRAM {summary.vram_peak_pct:.1f}%). Neither compute nor host CPU is saturated."
        ),
        recommendations=[
            "Increase concurrency above 30 or verify vLLM --max-num-seqs is >= 30.",
            "Check network latency between benchmark client and vLLM server.",
        ],
    )
