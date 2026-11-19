<#
.SYNOPSIS
  Open SSH Tunnel to Grafana & Prometheus on the GPU server (Windows PowerShell)

.EXAMPLE
  .\scripts\connect_tunnel.ps1
  .\scripts\connect_tunnel.ps1 -Target user@tensor-node01
#>
param(
    [string]$Target = "",
    [int]$SshPort = 22,
    [int]$LocalGrafanaPort = 3000,
    [int]$RemoteGrafanaPort = 3000,
    [int]$LocalPrometheusPort = 9090,
    [int]$RemotePrometheusPort = 9090
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MonDir = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $MonDir ".env"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim()
            if ($k -eq "GPU_SSH_TARGET" -and [string]::IsNullOrWhiteSpace($Target)) { $Target = $v }
            if ($k -eq "SSH_PORT") { $SshPort = [int]$v }
            if ($k -eq "GRAFANA_PORT") { $RemoteGrafanaPort = [int]$v }
            if ($k -eq "PROMETHEUS_PORT") { $RemotePrometheusPort = [int]$v }
            if ($k -eq "LOCAL_GRAFANA_PORT") { $LocalGrafanaPort = [int]$v }
            if ($k -eq "LOCAL_PROMETHEUS_PORT") { $LocalPrometheusPort = [int]$v }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($Target)) {
    $Target = "user@tensor-node01"
}

Write-Host "Opening SSH tunnel to ${Target} (port ${SshPort})..." -ForegroundColor Cyan
Write-Host "  Grafana    -> http://localhost:${LocalGrafanaPort}" -ForegroundColor Green
Write-Host "  Prometheus -> http://localhost:${LocalPrometheusPort}" -ForegroundColor Green
Write-Host "Press Ctrl+C to close the tunnel." -ForegroundColor Yellow
Write-Host ""

ssh -N -p $SshPort `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 `
    -L "${LocalGrafanaPort}:localhost:${RemoteGrafanaPort}" `
    -L "${LocalPrometheusPort}:localhost:${RemotePrometheusPort}" `
    $Target
