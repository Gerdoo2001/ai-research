$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunId = "fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42"
$SmokeId = "smoke_ft007_densenet201_label_smoothing005_s42"
$RunDir = Join-Path $RepoRoot "model\results\runs\$RunId"
$SmokeDir = Join-Path $RepoRoot "model\results\runs\$SmokeId"
$Host.UI.RawUI.WindowTitle = "LIVE FT-007 | DenseNet-201 Label Smoothing 0.05"

while ($true) {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "LIVE FT-007: DenseNet-201 + VIRIDIS -> GE -> CLAHE + LABEL SMOOTHING" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Only change: label smoothing 0.00 -> 0.05"
    Write-Host "Loss: balanced class-weighted cross-entropy"
    Write-Host "Controls: 224px | batch 8 | AdamW | LR 0.0001 | WD 0.0001 | seed 42"
    Write-Host "Reference: F1 85.294% | balanced accuracy 84.546% | accuracy 86.613% | Viral recall 62.332%"
    Write-Host "Locked test: untouched"
    Write-Host ""

    $ActiveDir = $null
    $Label = $null
    if (Test-Path (Join-Path $RunDir "status.json")) {
        $ActiveDir = $RunDir
        $Label = "FULL SEED-42 TRAINING"
    } elseif (Test-Path (Join-Path $SmokeDir "status.json")) {
        $ActiveDir = $SmokeDir
        $Label = "LIMITED SMOKE TEST (NOT AN EFFECTIVENESS RESULT)"
    }

    if ($null -eq $ActiveDir) {
        Write-Host "Waiting for contract tests and smoke initialization..." -ForegroundColor Yellow
    } else {
        $StatusPath = Join-Path $ActiveDir "status.json"
        $LivePath = Join-Path $ActiveDir "live_progress.json"
        $HistoryPath = Join-Path $ActiveDir "history.csv"
        $S = Get-Content -Raw $StatusPath | ConvertFrom-Json
        Write-Host $Label -ForegroundColor Green
        Write-Host ("Status:            {0}" -f $S.status)
        Write-Host ("Epoch:             {0}/{1}" -f $S.current_epoch, $S.total_epochs)
        Write-Host ("Best epoch:        {0}" -f $S.best_epoch)
        if ($null -ne $S.best_validation_macro_f1) {
            Write-Host ("Best val macro F1: {0:N3}%" -f (100.0 * $S.best_validation_macro_f1))
        }
        if (Test-Path $LivePath) {
            $L = Get-Content -Raw $LivePath | ConvertFrom-Json
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
        if (Test-Path $HistoryPath) {
            $Last = Import-Csv $HistoryPath | Select-Object -Last 1
            if ($null -ne $Last) {
                Write-Host ""
                Write-Host "LATEST COMPLETED EPOCH" -ForegroundColor Magenta
                Write-Host ("Validation F1:     {0:N3}%" -f (100.0 * [double]$Last.val_macro_f1))
                Write-Host ("Balanced accuracy: {0:N3}%" -f (100.0 * [double]$Last.val_balanced_accuracy))
                Write-Host ("Accuracy:          {0:N3}%" -f (100.0 * [double]$Last.val_accuracy))
                Write-Host ("Viral recall:      {0:N3}%" -f (100.0 * [double]$Last.recall_viral_or_pneumonia))
                Write-Host ("Epoch duration:    {0:N1} minutes" -f ([double]$Last.elapsed_seconds / 60.0))
            }
        }
    }
    Write-Host ""
    Write-Host ("Last refresh: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor DarkGray
    Write-Host "Closing this monitor does not stop training." -ForegroundColor DarkGray
    Start-Sleep -Seconds 3
}
