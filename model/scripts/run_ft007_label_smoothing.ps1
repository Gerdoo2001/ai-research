$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Config = Join-Path $RepoRoot "model\configs\fine_tuning\densenet201_viridis_ge_clahe_label_smoothing005_s42.json"
$RunId = "fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42"
$RunDir = Join-Path $RepoRoot "model\results\runs\$RunId"
$StatusPath = Join-Path $RunDir "status.json"

Set-Location $RepoRoot
$Host.UI.RawUI.WindowTitle = "FT-007 | DenseNet-201 | Cross-Entropy + Label Smoothing 0.05"
Clear-Host
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "FT-007 CLEAN LABEL-SMOOTHING CLASSIFICATION ABLATION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Architecture:       DenseNet-201, ImageNet pretrained"
Write-Host "Preprocessing:      VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE"
Write-Host "Only changed field: label smoothing 0.00 -> 0.05"
Write-Host "Loss:               balanced class-weighted cross-entropy"
Write-Host "Class weighting:    balanced, derived from locked training counts"
Write-Host "Input / batch:      224 x 224 / 8"
Write-Host "Optimizer:          AdamW; LR=0.0001; weight decay=0.0001"
Write-Host "Scheduler:          ReduceLROnPlateau; factor=0.5; patience=2"
Write-Host "Augmentation:       horizontal flip p=0.5; rotation +/-5 degrees"
Write-Host "Training budget:    maximum 12 epochs; early-stop patience=4"
Write-Host "Seed:               42"
Write-Host "Primary metric:     validation macro F1"
Write-Host "Reference F1:       85.294%"
Write-Host "Locked test:        NOT opened or evaluated"
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Running implementation and historical-hash contracts..." -ForegroundColor Yellow
& $Python -m unittest model.tests.test_multiclass_pipeline.ConfigTests model.tests.test_multiclass_pipeline.ManifestTests model.tests.test_multiclass_pipeline.TransformTests model.tests.test_multiclass_pipeline.MetricTests -v
if ($LASTEXITCODE -ne 0) { throw "FT-007 contract tests failed." }

Write-Host "`n[2/3] Running 12-train / 12-validation ImageNet smoke test..." -ForegroundColor Yellow
& $Python "model\scripts\train_multiclass.py" --config $Config --smoke-test --weights imagenet --run-id "smoke_ft007_densenet201_label_smoothing005_s42"
if ($LASTEXITCODE -ne 0) { throw "FT-007 smoke test failed; full training was not started." }

Write-Host "`n[3/3] Starting/resuming the full seed-42 label-smoothing run..." -ForegroundColor Green
$ResumeArguments = @()
if (Test-Path $StatusPath) {
    $ExistingStatus = Get-Content -Raw $StatusPath | ConvertFrom-Json
    if ($ExistingStatus.status -eq "completed") {
        Write-Host "FT-007 is already complete; no duplicate run was launched." -ForegroundColor Green
        exit 0
    }
    if ((Test-Path (Join-Path $RunDir "last_checkpoint.pt")) -and $ExistingStatus.current_epoch -gt 0) {
        $ResumeArguments = @("--resume")
        Write-Host "Resuming from the checkpoint after epoch $($ExistingStatus.current_epoch)." -ForegroundColor Yellow
    }
}
& $Python "model\scripts\train_multiclass.py" --config $Config @ResumeArguments
if ($LASTEXITCODE -ne 0) { throw "FT-007 failed or was interrupted; saved checkpoints remain available." }

Write-Host "`nFT-007 completed. Artifacts:" -ForegroundColor Green
Write-Host $RunDir
Write-Host "The locked test remains untouched." -ForegroundColor Green
