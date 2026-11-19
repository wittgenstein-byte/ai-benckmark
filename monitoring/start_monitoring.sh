#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[Monitoring] Starting Prometheus, Grafana, Node Exporter, and NVIDIA DCGM Exporter..."
docker compose up -d

echo ""
echo "Monitoring stack is running:"
echo "  - Grafana Dashboard : http://localhost:3000  (admin / admin)"
echo "    -> Folder: 'vLLM Benchmark' -> 'vLLM RTX 6000 Benchmark & Bottleneck Monitor'"
echo "  - Prometheus Targets: http://localhost:9090/targets"
echo "  - DCGM Metrics      : http://localhost:9400/metrics"
echo "  - Node Metrics      : http://localhost:9100/metrics"
