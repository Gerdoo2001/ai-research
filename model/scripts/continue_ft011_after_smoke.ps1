param([Parameter(Mandatory = $true)][int]$SmokeProcessId)
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$experimentPath = Join-Path $repoRoot 'model\results\ft011'
$smokeVerified = Join-Path $experimentPath 'smoke_verified.json'
$logPath = Join-Path $experimentPath 'queue_console.log'
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$SmokeProcessId"
if ($process -and ($process.Name -notmatch '^python\.exe$' -or $process.CommandLine -notmatch 'run_ft011\.py"?\s+--smoke(?:\s|$)')) {
    throw 'The supplied process is not the current FT-011 GPU smoke job.'
}
Set-Location -LiteralPath $repoRoot
while (Get-Process -Id $SmokeProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 2
}
if (-not (Test-Path -LiteralPath $smokeVerified -PathType Leaf)) {
    throw 'The smoke process ended without a verified protected smoke result. Full training was not started.'
}
& $pythonPath -u (Join-Path $PSScriptRoot 'run_ft011.py') --run *>> $logPath
if ($LASTEXITCODE -ne 0) { throw "FT-011 queue stopped with exit code $LASTEXITCODE. See $logPath" }
