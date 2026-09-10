$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "FT-008A2 | Corrected Bilateral U-Net Lung ROI Audit"
Set-Location $RepoRoot
Clear-Host
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "FT-008A2 CORRECTED BILATERAL U-NET LUNG-SEGMENTATION AUDIT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Dependency:         FT-007 label-smoothing training must finish first"
Write-Host "Model A:            Classic U-Net + cxr_reg_weights.best.hdf5"
Write-Host "Model B:            Legacy complete U-Net + trained_model.hdf5"
Write-Host "Screening cohort:   Same 90 validation images used in FT-002C"
Write-Host "Frozen policy:      grayscale 256x256; threshold 0.50; retain two largest regions; bbox margin 18%"
Write-Host "Output:             original + two mask/crop overlays per image"
Write-Host "Classifier:         NOT queued; manual review is mandatory first"
Write-Host "Locked test:        NOT opened or evaluated"
Write-Host "================================================================================" -ForegroundColor Cyan
& $Python "model\scripts\run_ft008_unet_queue.py"
if ($LASTEXITCODE -ne 0) { throw "FT-008A2 queue stopped with exit code $LASTEXITCODE" }
Write-Host "`nFT-008A2 corrected screening is ready for manual review." -ForegroundColor Green
Write-Host "No ROI classifier was started." -ForegroundColor Green
