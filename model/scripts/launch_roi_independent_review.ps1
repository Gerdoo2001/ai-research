$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'Independent ROI Audit - Frozen PSPNet 0.20 + 18 Percent Margin'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
$env:PYTHONUTF8 = '1'
Write-Host ('=' * 108) -ForegroundColor Cyan
Write-Host 'INDEPENDENT 90-IMAGE PSPNET ROI AUDIT' -ForegroundColor Cyan
Write-Host 'These are 90 new validation images: 30 Normal, 30 Viral, and 30 Bacterial.'
Write-Host 'There is zero overlap with the first 90-image audit.'
Write-Host 'The policy is frozen: PSPNet threshold 0.20 and crop margin 18%.' -ForegroundColor Yellow
Write-Host ''
Write-Host '  Y = Both lung masks and the proposed crop are acceptable'
Write-Host '  N = A mask or the crop is unacceptable'
Write-Host '  B = Go back one image'
Write-Host '  Q = Save and pause'
Write-Host ''
Write-Host 'Every answer is saved immediately. The locked test remains untouched.' -ForegroundColor Green
Write-Host ('=' * 108) -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -u model\scripts\review_roi_independent_gui.py
Write-Host ''
Write-Host 'The independent review has closed. All decisions remain saved.' -ForegroundColor Green
Write-Host 'You may close this terminal or press Enter.'
Read-Host | Out-Null
