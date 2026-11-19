#!/usr/bin/env bash
set -euo pipefail

# Install & Start GPU + vLLM Monitoring Stack (DCGM + Node Exporter + Prometheus + Grafana)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${MON_DIR}"

echo "=== Step 1: Checking Prerequisites ==="
for cmd in docker curl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: '${cmd}' is not installed." >&2
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: 'docker compose' plugin is not available." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "WARNING: 'nvidia-smi' not found. DCGM Exporter requires an NVIDIA GPU & driver."
fi

if ! docker info 2>/dev/null | grep -qi "nvidia"; then
  echo "WARNING: NVIDIA Container Runtime not obvious in 'docker info'. If dcgm-exporter fails, verify nvidia-container-toolkit."
fi

echo ""
echo "=== Step 2: Preparing .env ==="
if [[ ! -f "${MON_DIR}/.env" ]]; then
  cp "${MON_DIR}/.env.example" "${MON_DIR}/.env"
  echo "Created .env from .env.example (edit .env if you need custom ports)."
else
  echo "Using existing .env"
fi

set -a
# shellcheck disable=SC1091
source "${MON_DIR}/.env"
set +a

chmod +x "${SCRIPT_DIR}"/*.sh
# Ensure Grafana (UID 472) and Prometheus (UID 65534) containers can read mounted configs/dashboards
chmod -R a+rX "${MON_DIR}/grafana" "${MON_DIR}/prometheus" 2>/dev/null || true

echo ""
echo "=== Step 3: Pre-Flight Port & Outbound Check ==="
"${SCRIPT_DIR}/check_ports.sh" --pre

echo ""
echo "=== Step 4: Starting Docker Compose Stack ==="
docker compose -p benchmark-monitoring up -d

echo ""
echo "Waiting for Grafana and DCGM Exporter to become ready (up to 30s)..."
for i in $(seq 1 15); do
  if curl -sf --connect-timeout 2 "http://localhost:${GRAFANA_PORT:-3000}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo ""
echo "=== Step 5: Post-Up Verification ==="
"${SCRIPT_DIR}/check_ports.sh" --post

echo ""
echo "============================================================"
echo "Monitoring Stack is UP!"
echo "  Grafana       : http://localhost:${GRAFANA_PORT:-3000} (user: ${GRAFANA_ADMIN_USER:-admin})"
echo "  Prometheus    : http://localhost:${PROMETHEUS_PORT:-9090}"
echo "  DCGM Exporter : http://localhost:${DCGM_PORT:-9400}/metrics"
echo "  Node Exporter : http://localhost:${NODE_EXPORTER_PORT:-9100}/metrics"
echo ""
echo "If accessing from your laptop, open an SSH tunnel:"
echo "  Windows: .\\scripts\\connect_tunnel.ps1"
echo "  Mac/WSL: ./scripts/connect_tunnel.sh"
echo "============================================================"
