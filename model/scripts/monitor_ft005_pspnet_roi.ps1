$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Precompute = Join-Path $RepoRoot "model\results\roi_pspnet_bbox_precompute\progress.json"
$RunId = "fine_tuning_densenet201_viridis_ge_clahe_pspnet_roi_s42"
$RunDir = Join-Path $RepoRoot "model\results\runs\$RunId"
$Live = Join-Path $RunDir "live_progress.json"
$Status = Join-Path $RunDir "status.json"
$History = Join-Path $RunDir "history.csv"

$Host.UI.RawUI.WindowTitle = "LIVE FT-005 | DenseNet-201 + PSPNet ROI"
while ($true) {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "LIVE FT-005: DenseNet-201 + PSPNet ROI + VIRIDIS -> GE -> CLAHE" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Only variable versus the reference winner: frozen PSPNet lung-centered crop"
    Write-Host "Comparison metric: validation macro F1 | Seed: 42 | Locked test: untouched"
    Write-Host ""

    if (Test-Path $Precompute) {
        $P = Get-Content -Raw $Precompute | ConvertFrom-Json
        $Percent = if ($P.total_images -gt 0) { 100.0 * $P.completed_images / $P.total_images } else { 0 }
        Write-Host "ROI BOUNDING-BOX PRECOMPUTATION" -ForegroundColor Yellow
        Write-Host ("Status:            {0}" -f $P.status)
        Write-Host ("Completed:         {0}/{1} ({2:N2}%)" -f $P.completed_images, $P.total_images, $Percent)
        Write-Host ("Current split:     {0}" -f $P.current_split)
        Write-Host ("Current image:     {0}" -f $P.current_image)
        if ($null -ne $P.estimated_remaining_seconds) {
            Write-Host ("Estimated time:    {0:N1} minutes" -f ($P.estimated_remaining_seconds / 60.0))
        }
        Write-Host ""
    } else {
        Write-Host "ROI precomputation is waiting to start." -ForegroundColor DarkYellow
        Write-Host ""
    }

    if (Test-Path $Status) {
        $S = Get-Content -Raw $Status | ConvertFrom-Json
        Write-Host "FULL CLASSIFIER TRAINING" -ForegroundColor Green
        Write-Host ("Status:            {0}" -f $S.status)
        Write-Host ("Epoch:             {0}/{1}" -f $S.current_epoch, $S.total_epochs)
        Write-Host ("Best epoch:        {0}" -f $S.best_epoch)
        if ($null -ne $S.best_validation_macro_f1) {
            Write-Host ("Best val macro F1: {0:N3}%" -f (100.0 * $S.best_validation_macro_f1))
        }
    }
    if (Test-Path $Live) {
        $L = Get-Content -Raw $Live | ConvertFrom-Json
        Write-Host ("Current stage:     {0}" -f $L.stage)
        if ($L.total_batches -gt 0) {
            $Width = 40
            $Filled = [Math]::Floor($Width * $L.fraction)
            $Bar = ("#" * $Filled) + ("-" * ($Width - $Filled))
            Write-Host ("Current epoch:     [{0}] {1:N1}% ({2}/{3} batches)" -f $Bar, (100.0 * $L.fraction), $L.batch, $L.total_batches)
        }
        if ($null -ne $L.running_train_loss) {
            Write-Host ("Running loss:      {0:N4}" -f $L.running_train_loss)
        }
        if ($null -ne $L.estimated_remaining_seconds) {
            Write-Host ("Epoch time left:   {0:N1} minutes" -f ($L.estimated_remaining_seconds / 60.0))
        }
    }
    if (Test-Path $History) {
        $Last = Import-Csv $History | Select-Object -Last 1
        if ($null -ne $Last) {
            Write-Host ""
            Write-Host "LATEST COMPLETED EPOCH" -ForegroundColor Magenta
            Write-Host ("Epoch:             {0}" -f $Last.epoch)
            Write-Host ("Validation F1:     {0:N3}%" -f (100.0 * [double]$Last.val_macro_f1))
            Write-Host ("Balanced accuracy: {0:N3}%" -f (100.0 * [double]$Last.val_balanced_accuracy))
            Write-Host ("Accuracy:          {0:N3}%" -f (100.0 * [double]$Last.val_accuracy))
            Write-Host ("Viral recall:      {0:N3}%" -f (100.0 * [double]$Last.recall_viral_or_pneumonia))
            Write-Host ("Epoch duration:    {0:N1} minutes" -f ([double]$Last.elapsed_seconds / 60.0))
        }
    }
    Write-Host ""
    Write-Host ("Last screen refresh: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor DarkGray
    Write-Host "Close this monitor window at any time; it does not stop the experiment." -ForegroundColor DarkGray
    Start-Sleep -Seconds 3
}
