param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Config = Join-Path $RepoRoot "model\configs\fine_tuning\densenet201_viridis_ge_clahe_pspnet_roi_s42.json"
$RunId = "fine_tuning_densenet201_viridis_ge_clahe_pspnet_roi_s42"
$RunDir = Join-Path $RepoRoot "model\results\runs\$RunId"
$StatusPath = Join-Path $RunDir "status.json"

Set-Location $RepoRoot
$Host.UI.RawUI.WindowTitle = "FT-005 | DenseNet-201 | PSPNet ROI | VIRIDIS -> GE -> CLAHE"
Clear-Host

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "FT-005 CONTROLLED FINE-TUNING EXPERIMENT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Classifier:        DenseNet-201 (ImageNet pretrained)"
Write-Host "New variable:      Frozen TorchXRayVision ChestX-Det PSPNet lung ROI crop"
Write-Host "ROI threshold:     0.20"
Write-Host "ROI margin:        18% around union of left/right lung masks"
Write-Host "Enhancement order: VIRIDIS pseudo-color -> Global Histogram Equalization (GE) -> CLAHE"
Write-Host "Input:             224 x 224"
Write-Host "Augmentation:      Horizontal flip p=0.5; random rotation +/-5 degrees (training only)"
Write-Host "Optimization:      AdamW; LR=0.0001; weight decay=0.0001; balanced cross-entropy"
Write-Host "Schedule:          ReduceLROnPlateau; factor=0.5; patience=2"
Write-Host "Stopping:          Maximum 12 epochs; early-stopping patience=4; min delta=0.0001"
Write-Host "Batch / seed:      8 / 42"
Write-Host "Evidence split:    Locked training + validation manifests only"
Write-Host "Locked test:       NOT opened, evaluated, or used for model selection"
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Precomputing deterministic ROI bounding boxes (restartable)..." -ForegroundColor Yellow
& $Python "model\scripts\build_pspnet_roi_bbox_manifest.py" --device cuda
if ($LASTEXITCODE -ne 0) { throw "ROI bbox precomputation failed." }

if (-not $SkipTests) {
    Write-Host "`n[2/4] Running implementation contract tests..." -ForegroundColor Yellow
    & $Python -m unittest model.tests.test_multiclass_pipeline.ConfigTests model.tests.test_multiclass_pipeline.ManifestTests model.tests.test_multiclass_pipeline.TransformTests -v
    if ($LASTEXITCODE -ne 0) { throw "FT-005 contract tests failed." }
} else {
    Write-Host "`n[2/4] Contract tests skipped by explicit switch." -ForegroundColor DarkYellow
}

Write-Host "`n[3/4] Running a 12-train / 12-validation smoke test..." -ForegroundColor Yellow
& $Python "model\scripts\train_multiclass.py" --config $Config --smoke-test --weights imagenet --run-id "smoke_ft005_densenet201_pspnet_roi_s42"
if ($LASTEXITCODE -ne 0) { throw "FT-005 smoke test failed; full training was not started." }

Write-Host "`n[4/4] Starting/resuming the full controlled seed-42 run..." -ForegroundColor Green
$ResumeArguments = @()
if (Test-Path $StatusPath) {
    $ExistingStatus = Get-Content -Raw $StatusPath | ConvertFrom-Json
    if ($ExistingStatus.status -eq "completed") {
        Write-Host "The full FT-005 run is already complete; no duplicate run was launched." -ForegroundColor Green
        exit 0
    }
    if ((Test-Path (Join-Path $RunDir "last_checkpoint.pt")) -and $ExistingStatus.current_epoch -gt 0) {
        $ResumeArguments = @("--resume")
        Write-Host "A compatible checkpoint exists; resuming after epoch $($ExistingStatus.current_epoch)." -ForegroundColor Yellow
    }
}
& $Python "model\scripts\train_multiclass.py" --config $Config @ResumeArguments
if ($LASTEXITCODE -ne 0) { throw "FT-005 full training failed or was interrupted; persisted artifacts remain resumable." }

Write-Host "`nFT-005 completed. Validation metrics and artifacts are in:" -ForegroundColor Green
Write-Host $RunDir
Write-Host "The locked test remains untouched." -ForegroundColor Green
