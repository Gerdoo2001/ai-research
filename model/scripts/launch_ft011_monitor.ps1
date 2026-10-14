param(
    [Parameter(Mandatory = $true)][string]$LogDir,
    [string]$ProgressPath = '',
    [ValidateSet('overview', 'config', 'results', 'hardware', 'files', 'all')][string]$View = 'overview'
)
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$monitorPath = Join-Path $PSScriptRoot 'monitor_ft011.py'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Missing research Python: $pythonPath" }
$resolvedLog = [System.IO.Path]::GetFullPath($LogDir)
$launchArguments = @(('"' + $monitorPath + '"'), '--log-dir', ('"' + $resolvedLog + '"'), '--view', $View)
if ($ProgressPath) {
    $resolvedProgress = [System.IO.Path]::GetFullPath($ProgressPath)
    $launchArguments += @('--progress', ('"' + $resolvedProgress + '"'))
}
# The visible console is explicitly requested. The dashboard sets its title.
# Start directly: this avoids a nested PowerShell host and quoted-command issues.
Start-Process -FilePath $pythonPath -WorkingDirectory $repoRoot -WindowStyle Normal -ArgumentList $launchArguments -PassThru

