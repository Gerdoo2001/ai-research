$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'ROI Manual Review - Y Yes, N No, B Back, Q Pause'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
$env:PYTHONUTF8 = '1'
Write-Host ('=' * 100) -ForegroundColor Cyan
Write-Host 'PSPNET LUNG ROI MANUAL REVIEW' -ForegroundColor Cyan
Write-Host 'The image window accepts: Y = acceptable, N = unacceptable, B = previous, Q = save and pause.'
Write-Host 'Every response is written immediately to the persistent review record.'
Write-Host ('=' * 100) -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -u model\scripts\review_roi_masks_gui.py
Write-Host ''
Write-Host 'The review program has closed. All completed decisions remain saved.' -ForegroundColor Green
Write-Host 'You may close this terminal or press Enter.'
Read-Host | Out-Null
