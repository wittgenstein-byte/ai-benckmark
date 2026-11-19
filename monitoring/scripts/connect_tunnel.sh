#!/usr/bin/env bash
set -euo pipefail

# Open SSH Tunnel to Grafana & Prometheus on the GPU server (Linux/Mac/WSL)
# Usage:
#   ./scripts/connect_tunnel.sh
#   GPU_SSH_TARGET=user@tensor-node01 ./scripts/connect_tunnel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${MON_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${MON_DIR}/.env"
  set +a
fi

TARGET="${1:-${GPU_SSH_TARGET:-user@tensor-node01}}"
SSH_PORT="${SSH_PORT:-22}"
LOCAL_GRAFANA_PORT="${LOCAL_GRAFANA_PORT:-3000}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
LOCAL_PROMETHEUS_PORT="${LOCAL_PROMETHEUS_PORT:-9090}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"

echo "Opening SSH tunnel to ${TARGET} (port ${SSH_PORT})..."
echo "  Grafana    -> http://localhost:${LOCAL_GRAFANA_PORT}"
echo "  Prometheus -> http://localhost:${LOCAL_PROMETHEUS_PORT}"
echo "Press Ctrl+C to close the tunnel."
echo ""

exec ssh -N -p "${SSH_PORT}" \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "${LOCAL_GRAFANA_PORT}:localhost:${GRAFANA_PORT}" \
  -L "${LOCAL_PROMETHEUS_PORT}:localhost:${PROMETHEUS_PORT}" \
  "${TARGET}"
