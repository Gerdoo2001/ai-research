param()

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "model\scripts\run_ft009b_hierarchical.py"
$OutputDir = Join-Path $RepoRoot "model\results\ft009b_hierarchical_s42"
$Stdout = Join-Path $OutputDir "orchestrator_stdout.log"
$Stderr = Join-Path $OutputDir "orchestrator_stderr.log"
$ProcessRecord = Join-Path $OutputDir "orchestrator_process.json"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*run_ft009b_hierarchical.py*" -and $_.ProcessId -ne $PID
}
if ($existing) {
    throw "FT-009B already appears to be running under PID $($existing[0].ProcessId)."
}

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(("`"{0}`"" -f $Runner)) `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

@{
    experiment_id = "FT-009B-HIERARCHICAL-S42"
    process_id = $process.Id
    started_at = (Get-Date).ToString("o")
    python = $Python
    runner = $Runner
    stdout = $Stdout
    stderr = $Stderr
    locked_test_accessed = $false
} | ConvertTo-Json | Set-Content -Encoding UTF8 $ProcessRecord

Write-Output "FT-009B seed-42 orchestrator started."
Write-Output "PID: $($process.Id)"
Write-Output "Process record: $ProcessRecord"
Write-Output "Standard output: $Stdout"
Write-Output "Standard error: $Stderr"
