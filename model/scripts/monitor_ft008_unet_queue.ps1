$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$QueuePath = Join-Path $RepoRoot "model\results\ft008a2_unet_queue\progress.json"
$AuditPath = Join-Path $RepoRoot "model\results\roi_unet_screening_two_components\progress.json"
$Ft007Dir = Join-Path $RepoRoot "model\results\runs\fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42"
$Host.UI.RawUI.WindowTitle = "LIVE FT-008A2 | Corrected Bilateral U-Net Audit"

while ($true) {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "LIVE FT-008A2: CORRECTED TWO-MODEL BILATERAL U-NET AUDIT" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Model A: Classic U-Net | weights: cxr_reg_weights.best.hdf5"
    Write-Host "Model B: Legacy full U-Net | model: trained_model.hdf5"
    Write-Host "Policy:  center-square grayscale -> 256x256 -> threshold 0.50 -> two largest lung regions -> 18% crop margin"
    Write-Host "Scope:   90 validation images (30 Normal, 30 Viral, 30 Bacterial), same cohort as FT-002C"
    Write-Host "Safety:  no locked test; no classifier training; manual review required"
    Write-Host ""

    $FtStatusPath = Join-Path $Ft007Dir "status.json"
    if (Test-Path $FtStatusPath) {
        $F = Get-Content -Raw $FtStatusPath | ConvertFrom-Json
        Write-Host "DEPENDENCY: FT-007 LABEL SMOOTHING" -ForegroundColor Magenta
        Write-Host ("Status:            {0}" -f $F.status)
        Write-Host ("Completed epochs:  {0}/{1}" -f $F.current_epoch, $F.total_epochs)
        if ($null -ne $F.best_validation_macro_f1) {
            Write-Host ("Best val macro F1: {0:N3}%" -f (100.0 * $F.best_validation_macro_f1))
        }
        $LivePath = Join-Path $Ft007Dir "live_progress.json"
        if ((Test-Path $LivePath) -and $F.status -eq "running") {
            $L = Get-Content -Raw $LivePath | ConvertFrom-Json
            if ($L.total_batches -gt 0) {
                Write-Host ("Current epoch:     {0}/{1} | {2:N1}% | batch {3}/{4}" -f $L.epoch, $L.total_epochs, (100.0*$L.fraction), $L.batch, $L.total_batches)
            }
        }
    }
    Write-Host ""
    Write-Host "FT-008A2 QUEUE" -ForegroundColor Green
    if (Test-Path $QueuePath) {
        $Q = Get-Content -Raw $QueuePath | ConvertFrom-Json
        Write-Host ("Queue status:       {0}" -f $Q.status)
        if ($Q.failure_reason) { Write-Host ("Failure:            {0}" -f $Q.failure_reason) -ForegroundColor Red }
    } else {
        Write-Host "Queue status:       initializing"
    }
    if (Test-Path $AuditPath) {
        $A = Get-Content -Raw $AuditPath | ConvertFrom-Json
        Write-Host ("Audit status:       {0}" -f $A.status)
        Write-Host ("Images completed:   {0}/{1}" -f $A.completed_images, $A.total_images)
        if ($A.current_class) { Write-Host ("Current class:      {0}" -f $A.current_class) }
        if ($A.current_image) { Write-Host ("Current image:      {0}" -f $A.current_image) }
        if ($null -ne $A.estimated_remaining_seconds) {
            Write-Host ("Estimated time:     {0:N1} minutes" -f ($A.estimated_remaining_seconds / 60.0))
        }
        if ($A.per_model) {
            Write-Host ""
            Write-Host "AUTOMATED SCREENING COUNTS (NOT ANATOMICAL ACCURACY)" -ForegroundColor Yellow
            foreach ($property in $A.per_model.PSObject.Properties) {
                $M = $property.Value
                Write-Host ("{0}: passes={1} flags={2} empty={3}" -f $property.Name, $M.automated_passes, $M.automated_flags, $M.empty_masks)
            }
        }
    }
    Write-Host ""
    Write-Host ("Last refresh: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor DarkGray
    Write-Host "Closing this monitor does not stop the queue." -ForegroundColor DarkGray
    Start-Sleep -Seconds 3
}
