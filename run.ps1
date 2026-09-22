# Runs a sweep over alpha x seed, one flwr run after another; each starts only
# when the previous one has finished. Stops at the first failure.
#
# Usage (venv active):   .\run_grid.ps1

# ---- EDIT THESE ------------------------------------------------------------
# Folder that contains the app's pyproject.toml (the runs execute here):
$ProjectPath = 'C:\Users\lauti\Desktop\programas\Federated-Learning\Aggregation_Strategies\FedProx'

$alphas = @(0.1, 0.5, 1, 1000)
$seeds  = @(27, 42, 143)
$mu     = 0.01
# ----------------------------------------------------------------------------

# Outer loop = alpha, inner loop = seed (12 runs with the values above).
$commands = @(
    foreach ($alpha in $alphas) {
        foreach ($seed in $seeds) {
            # "$var" interpolation always writes numbers with a '.' decimal point,
            # unlike the -f operator, which follows the PC's regional settings.
            "flwr run . --run-config `"alpha=$alpha seed=$seed mu=$mu`" --federation-config `"num-supernodes=10`" --stream"
        }
    }
)

# --- validate the target folder ---------------------------------------------
if (-not (Test-Path -LiteralPath $ProjectPath -PathType Container)) {
    Write-Host "Folder not found: $ProjectPath" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectPath "pyproject.toml"))) {
    Write-Host "No pyproject.toml in $ProjectPath - is this the app folder?" -ForegroundColor Red
    exit 1
}

# Logs go next to this script, regardless of where the runs execute.
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Start-Transcript -Path (Join-Path $logDir "grid_$(Get-Date -Format 'yyyyMMdd_HHmmss').log") | Out-Null

Push-Location -LiteralPath $ProjectPath
try {
    Write-Host "Running $($commands.Count) runs in: $ProjectPath" -ForegroundColor Yellow
    $i = 0
    foreach ($cmd in $commands) {
        $i++
        Write-Host "`n=== [$i/$($commands.Count)] $cmd ===" -ForegroundColor Cyan
        $start = Get-Date
        Invoke-Expression $cmd
        $exit = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
        if ($exit -ne 0) {
            Write-Host "FAILED (exit code $exit) after $mins min: $cmd" -ForegroundColor Red
            break
        }
        Write-Host "Done in $mins min" -ForegroundColor Green
    }
}
finally {
    Pop-Location
    Stop-Transcript | Out-Null
}