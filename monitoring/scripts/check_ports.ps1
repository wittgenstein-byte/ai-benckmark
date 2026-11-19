<#
.SYNOPSIS
  Client-side Outbound & Local Tunnel Port Checker (Windows PowerShell)

.DESCRIPTION
  Checks:
  1. Outbound TCP connectivity from your laptop to the GPU server (SSH, Grafana, Prometheus, vLLM).
  2. Local tunnel ports (localhost:3000, localhost:9090) if SSH tunnel is active.

.EXAMPLE
  .\scripts\check_ports.ps1 -RemoteHost tensor-node01
#>
param(
    [string]$RemoteHost = "",
    [int]$SshPort = 22,
    [int]$RemoteGrafanaPort = 3000,
    [int]$RemotePrometheusPort = 9090,
    [int]$RemoteVllmPort = 8000,
    [int]$LocalGrafanaPort = 3000,
    [int]$LocalPrometheusPort = 9090
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MonDir = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $MonDir ".env"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim()
            if ($k -eq "GPU_SSH_TARGET" -and [string]::IsNullOrWhiteSpace($RemoteHost)) {
                if ($v -match '@(.+)$') { $RemoteHost = $matches[1] } else { $RemoteHost = $v }
            }
            if ($k -eq "SSH_PORT") { $SshPort = [int]$v }
            if ($k -eq "GRAFANA_PORT") { $RemoteGrafanaPort = [int]$v }
            if ($k -eq "PROMETHEUS_PORT") { $RemotePrometheusPort = [int]$v }
            if ($k -eq "VLLM_PORT") { $RemoteVllmPort = [int]$v }
            if ($k -eq "LOCAL_GRAFANA_PORT") { $LocalGrafanaPort = [int]$v }
            if ($k -eq "LOCAL_PROMETHEUS_PORT") { $LocalPrometheusPort = [int]$v }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($RemoteHost)) {
    $RemoteHost = "tensor-node01"
}

function Test-TcpPort {
    param([string]$TargetHost, [int]$Port, [string]$Label)
    $res = Test-NetConnection -ComputerName $TargetHost -Port $Port -WarningAction SilentlyContinue
    if ($res.TcpTestSucceeded) {
        Write-Host ("  [OPEN]    {0}:{1} ({2})" -f $TargetHost, $Port, $Label) -ForegroundColor Green
    } else {
        Write-Host ("  [CLOSED]  {0}:{1} ({2})" -f $TargetHost, $Port, $Label) -ForegroundColor Yellow
    }
}

Write-Host "=== [Outbound from Laptop -> ${RemoteHost}] ===" -ForegroundColor Cyan
Test-TcpPort -TargetHost $RemoteHost -Port $SshPort -Label "SSH"
Test-TcpPort -TargetHost $RemoteHost -Port $RemoteGrafanaPort -Label "Grafana Direct"
Test-TcpPort -TargetHost $RemoteHost -Port $RemotePrometheusPort -Label "Prometheus Direct"
Test-TcpPort -TargetHost $RemoteHost -Port $RemoteVllmPort -Label "vLLM Direct"

Write-Host ""
Write-Host "=== [Local Tunnel Ports on localhost] ===" -ForegroundColor Cyan
Test-TcpPort -TargetHost "127.0.0.1" -Port $LocalGrafanaPort -Label "Grafana Tunnel"
Test-TcpPort -TargetHost "127.0.0.1" -Port $LocalPrometheusPort -Label "Prometheus Tunnel"

Write-Host ""
Write-Host "Tip: If Direct ports are [CLOSED] but SSH is [OPEN], run .\scripts\connect_tunnel.ps1" -ForegroundColor Gray
