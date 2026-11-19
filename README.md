# Healthcare AI & Local vLLM Benchmark Toolkit

End-to-end toolkit for exploring [Synthea](https://github.com/synthetichealth/synthea) EHR datasets (**CSV** / **FHIR R4 JSON**), compressing longitudinal patient records into token-efficient clinical summaries, benchmarking local **vLLM** models on **RTX 6000** GPUs (1,000 jobs @ 30 concurrent requests), and monitoring live GPU/CPU/RAM/KV-cache metrics in **Grafana**.

## Repository Structure

```text
.
├── synthea_eda.py              # CLI exploratory data analysis for Synthea CSV / FHIR R4
├── synthea_eda.ipynb           # Interactive EDA Jupyter notebook
├── synthea_etl.py              # Longitudinal patient ETL -> .md, .json, patients_summary.jsonl
├── run_benchmark.py            # CLI entrypoint for decoupled vLLM concurrency benchmarks
├── llm_benchmark/              # Modular benchmark engine (config, backends, workload, monitor, metrics, report)
├── configs/                    # Decoupled vLLM model profiles (JSON / YAML)
├── monitoring/                 # Docker Compose stack: Prometheus + Grafana + DCGM + Node Exporter
├── tests/                      # Unit & integration tests (ETL + Benchmark)
├── Dockerfile                  # Python 3.11 container for EDA/notebook
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data with Synthea (Optional)

If you do not already have Synthea output files, clone Synthea and generate a sample cohort with CSV export enabled:

```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./run_synthea -p 100 --exporter.csv.export=true
```

By default, Synthea writes outputs to:
- CSVs: `synthea/output/csv/`
- FHIR R4 Bundles: `synthea/output/fhir/`

### 3. Run the CLI EDA Script

```bash
# Auto-detect format (CSV or FHIR JSON)
python synthea_eda.py --input-dir path/to/synthea/output/csv --output-dir ./eda_output

# Explicitly specify format
python synthea_eda.py --input-dir path/to/synthea/output/fhir --format fhir --output-dir ./eda_output
```

Outputs written to `--output-dir`:
- `eda_report.md` — Summary statistics, missingness, top clinical codes, and cost breakdowns
- `figures/` — Publication-ready PNG charts (demographics, encounters, top conditions/medications, costs, and vitals)

### 4. Use the Interactive Notebook

```bash
jupyter notebook synthea_eda.ipynb
```

Update `INPUT_DIR` and `DATA_FORMAT` in the configuration cell and run all cells.

### 5. Patient Clinical Summary ETL (for GPU / LLM Inference)

Group raw Synthea CSV or FHIR R4 data by `PATIENT_ID`, deduplicate/compress longitudinal records, and export token-efficient `.md`, `.json`, and `.jsonl` summaries:

```bash
# Export per-patient .md, .json, and batch patients_summary.jsonl
python synthea_etl.py --input-dir path/to/synthea/output/csv --output-dir ./etl_output

# Optional: cap context window per patient and test on first 100 patients
python synthea_etl.py --input-dir path/to/synthea/output/csv --output-dir ./etl_output --max-tokens 2048 --limit 100
```

Outputs written to `--output-dir`:
- `md/<patient_id>.md` — Compact clinical summary in Markdown
- `json/<patient_id>.json` — Structured patient summary + `clinical_summary_md`
- `patients_summary.jsonl` — Single-file JSONL for batch GPU inference
- `etl_manifest.json` & `etl_report.md` — Compression ratio and estimated token distribution (`min`, `median`, `mean`, `p95`, `max`)

### 6. Local AI / vLLM Benchmark (RTX 6000 Series)

Benchmark a running vLLM OpenAI-compatible server with **1,000 jobs at 30 concurrent requests**, measuring:
- **Daily Capacity**: Projected `jobs/day`, `jobs/hr`, and `jobs/s`
- **Prompt Processing Speed**: Streaming **TTFT** (`min`, `p50`, `p95`, `p99`) and prefill `tok/s`
- **Generation Throughput**: System-wide output `tok/s` at 30 concurrent users, per-request decode `tok/s`, and TPOT/ITL
- **System Bottleneck**: Continuous CPU, RAM, and NVIDIA GPU (`nvidia-smi` compute % & VRAM %) sampling with automated bottleneck classification

#### Run with a Decoupled Model Config (Recommended)

Edit `configs/models.example.json` (or `.yaml`) to add/swap vLLM models without changing any code:

```bash
# Run a single model profile from config (1,000 jobs @ 30 concurrency)
python run_benchmark.py --config configs/models.example.json --models qwen-2.5-14b-awq

# Use Synthea ETL output as the real workload
python run_benchmark.py --config configs/models.example.json --models qwen-2.5-14b-awq \
  --prompt-source ./etl_output/patients_summary.jsonl --total-jobs 1000 --concurrency 30

# Compare multiple models sequentially
python run_benchmark.py --config configs/models.example.json --models llama-3.1-8b-fp8 qwen-2.5-14b-awq

# Quick ad-hoc run without a config file
python run_benchmark.py --model-id Qwen/Qwen2.5-14B-Instruct-AWQ --base-url http://localhost:8000/v1 \
  --total-jobs 1000 --concurrency 30

# Dry-run without a GPU (uses MockBackend)
python run_benchmark.py --config configs/models.example.json --models mock-dryrun --total-jobs 100 --concurrency 30
```

Outputs written to `--output-dir` (default `./benchmark_output`):
- `benchmark_report.md` — Side-by-side model comparison table & bottleneck diagnosis
- `benchmark_results.json` — Full structured telemetry and config
- `benchmark_jobs_<model>.csv` — Per-job TTFT, prefill/decode tok/s, and latency data

#### Run Unit Tests

```bash
python -m unittest discover -s tests -v
```

### 7. Grafana & Prometheus Live Dashboard (`monitoring/`)

A pre-provisioned monitoring stack (Prometheus + Grafana + Node Exporter + NVIDIA DCGM Exporter) is included under `monitoring/` with an auto-loaded dashboard (`vLLM RTX 6000 Benchmark & Bottleneck Monitor`):

```bash
cd monitoring
docker compose up -d
# Or: bash start_monitoring.sh
```

- **Grafana**: `http://localhost:3000` (`admin` / `admin`) → Dashboards → `vLLM Benchmark` → `vLLM RTX 6000 Benchmark & Bottleneck Monitor`
- **Prometheus Targets**: `http://localhost:9090/targets` (if vLLM runs on another host/port, update `monitoring/prometheus/prometheus.yml`)
