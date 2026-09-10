$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'ROI Recalibration - 1 Mild, 2 Moderate, 3 Aggressive, N None'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
$env:PYTHONUTF8 = '1'
Write-Host ('=' * 108) -ForegroundColor Cyan
Write-Host 'PSPNET LUNG ROI RECALIBRATION REVIEW' -ForegroundColor Cyan
Write-Host 'Compare the three expanded lung-mask/crop settings on the 21 images rejected in the first review.'
Write-Host ''
Write-Host '  1 = Mild:       threshold 0.40, crop margin 12%'
Write-Host '  2 = Moderate:   threshold 0.30, crop margin 15%'
Write-Host '  3 = Aggressive: threshold 0.20, crop margin 18%'
Write-Host '  N = None of the three is anatomically acceptable'
Write-Host '  B = Go back one image'
Write-Host '  Q = Save and pause'
Write-Host ''
Write-Host 'Every choice is saved immediately. This uses validation images only; the locked test remains untouched.' -ForegroundColor Yellow
Write-Host ('=' * 108) -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -u model\scripts\review_roi_recalibration_gui.py
Write-Host ''
Write-Host 'The review program has closed. All completed choices remain saved.' -ForegroundColor Green
Write-Host 'You may close this terminal or press Enter.'
Read-Host | Out-Null
