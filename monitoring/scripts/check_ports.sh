#!/usr/bin/env bash
set -euo pipefail

# Inbound & Outbound Port Checker on GPU Server
# Usage:
#   ./scripts/check_ports.sh --pre   # Before starting containers (check free ports & outbound)
#   ./scripts/check_ports.sh --post  # After starting containers (check listening & targets)
#   ./scripts/check_ports.sh         # Run both checks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:---all}"

if [[ -f "${MON_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${MON_DIR}/.env"
  set +a
fi

GRAFANA_PORT="${GRAFANA_PORT:-3000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
DCGM_PORT="${DCGM_PORT:-9400}"
NODE_EXPORTER_PORT="${NODE_EXPORTER_PORT:-9100}"
VLLM_PORT="${VLLM_PORT:-8000}"

PASS=0
WARN=0
FAIL=0

ok()   { echo "  [OK]   $*"; PASS=$((PASS + 1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN + 1)); }
err()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

check_port_listening() {
  local port="$1"
  local label="$2"
  local expect_state="$3" # "free" or "listening"

  local match=""
  if command -v ss >/dev/null 2>&1; then
    match="$(ss -tuln | awk -v p=":${port}$" '$5 ~ p {print $0}' || true)"
  elif command -v netstat >/dev/null 2>&1; then
    match="$(netstat -tuln | awk -v p=":${port}$" '$4 ~ p {print $0}' || true)"
  else
    warn "${label} (port ${port}): neither 'ss' nor 'netstat' found"
    return
  fi

  if [[ "${expect_state}" == "free" ]]; then
    if [[ -z "${match}" ]]; then
      ok "Port ${port} (${label}) is FREE"
    else
      err "Port ${port} (${label}) is ALREADY IN USE -> update .env to avoid collision!"
      echo "         ${match}"
    fi
  else
    if [[ -n "${match}" ]]; then
      ok "Port ${port} (${label}) is LISTENING"
      echo "         ${match}"
    else
      err "Port ${port} (${label}) is NOT listening"
    fi
  fi
}

check_firewall() {
  echo ""
  echo "=== [Inbound] Host Firewall Status ==="
  if command -v ufw >/dev/null 2>&1; then
    local ufw_out
    ufw_out="$(ufw status 2>/dev/null || true)"
    if [[ -n "${ufw_out}" ]]; then
      echo "${ufw_out}" | sed 's/^/  /'
    else
      warn "ufw installed but requires sudo to inspect. Use SSH tunnel if direct access fails."
    fi
  elif command -v firewall-cmd >/dev/null 2>&1; then
    local fw_state
    fw_state="$(firewall-cmd --state 2>/dev/null || true)"
    if [[ -n "${fw_state}" ]]; then
      ok "firewalld state: ${fw_state}"
    else
      warn "firewalld present (requires sudo to inspect)."
    fi
  else
    ok "No ufw/firewalld CLI detected (cloud/cluster firewall may still apply; SSH tunnel recommended)."
  fi
}

check_outbound() {
  echo ""
  echo "=== [Outbound] Registry & Local Target Reachability ==="
  for url in "https://registry-1.docker.io/v2/" "https://nvcr.io/v2/" "https://grafana.com"; do
    if curl -sS -o /dev/null --connect-timeout 5 -w "%{http_code}" "${url}" | grep -Eq "^(200|301|302|401)$"; then
      ok "Outbound HTTPS reachable: ${url}"
    else
      warn "Outbound HTTPS check failed or timed out: ${url} (needed for pulling images)"
    fi
  done

  if curl -sS -o /dev/null --connect-timeout 3 "http://localhost:${VLLM_PORT}/metrics"; then
    ok "vLLM metrics reachable at http://localhost:${VLLM_PORT}/metrics"
  else
    warn "vLLM metrics NOT reachable at http://localhost:${VLLM_PORT}/metrics (start vLLM if you want LLM panels)"
  fi
}

check_post_stack() {
  echo ""
  echo "=== [Post-Up] Service Listening & Health ==="
  check_port_listening "${GRAFANA_PORT}" "Grafana" "listening"
  check_port_listening "${PROMETHEUS_PORT}" "Prometheus" "listening"
  check_port_listening "${DCGM_PORT}" "DCGM Exporter" "listening"
  check_port_listening "${NODE_EXPORTER_PORT}" "Node Exporter" "listening"

  echo ""
  echo "=== [Post-Up] HTTP Health & Prometheus Targets ==="
  if curl -sf --connect-timeout 3 "http://localhost:${GRAFANA_PORT}/api/health" >/dev/null; then
    ok "Grafana /api/health responded 200 OK"
  else
    err "Grafana /api/health failed on port ${GRAFANA_PORT}"
  fi

  if curl -sf --connect-timeout 3 "http://localhost:${PROMETHEUS_PORT}/-/ready" >/dev/null; then
    ok "Prometheus /-/ready responded 200 OK"
  else
    err "Prometheus /-/ready failed on port ${PROMETHEUS_PORT}"
  fi

  if curl -sf --connect-timeout 3 "http://localhost:${DCGM_PORT}/metrics" | grep -q "DCGM_FI_DEV_"; then
    ok "DCGM Exporter exposing DCGM_FI_DEV_* metrics on port ${DCGM_PORT}"
  else
    warn "DCGM Exporter did not return DCGM_FI_DEV_* metrics (check 'docker logs benchmark-dcgm-exporter')"
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo ""
    echo "  Prometheus Scrape Targets:"
    curl -sf "http://localhost:${PROMETHEUS_PORT}/api/v1/targets" | python3 -c '
import sys, json
data = json.load(sys.stdin)
for t in data.get("data", {}).get("activeTargets", []):
    job = t.get("labels", {}).get("job")
    health = t.get("health")
    url = t.get("scrapeUrl")
    err = t.get("lastError", "")
    print(f"    - {job:15s} | {health:4s} | {url}" + (f" | err={err}" if err else ""))
' || warn "Could not parse Prometheus targets JSON"
  fi
}

echo "Monitoring Port & Connectivity Check (mode: ${MODE})"
echo "Ports -> Grafana:${GRAFANA_PORT}, Prometheus:${PROMETHEUS_PORT}, DCGM:${DCGM_PORT}, Node:${NODE_EXPORTER_PORT}, vLLM:${VLLM_PORT}"

if [[ "${MODE}" == "--pre" || "${MODE}" == "--all" ]]; then
  echo ""
  echo "=== [Pre-Flight] Checking Port Availability ==="
  check_port_listening "${GRAFANA_PORT}" "Grafana" "free"
  check_port_listening "${PROMETHEUS_PORT}" "Prometheus" "free"
  check_port_listening "${DCGM_PORT}" "DCGM Exporter" "free"
  check_port_listening "${NODE_EXPORTER_PORT}" "Node Exporter" "free"
  check_firewall
  check_outbound
fi

if [[ "${MODE}" == "--post" || "${MODE}" == "--all" ]]; then
  check_post_stack
fi

echo ""
echo "Summary: ${PASS} OK, ${WARN} WARN, ${FAIL} FAIL"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
