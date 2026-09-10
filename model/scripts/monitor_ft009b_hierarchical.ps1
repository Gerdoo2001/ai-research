param(
    [int]$RefreshSeconds = 3,
    [switch]$Once
)

$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProtocolPath = Join-Path $RepoRoot "model\configs\fine_tuning\ft009b_hierarchical_protocol.json"
$OutputDir = Join-Path $RepoRoot "model\results\ft009b_hierarchical_s42"
$ResultPath = Join-Path $OutputDir "ft009b_result.json"
$Stdout = Join-Path $OutputDir "orchestrator_stdout.log"
$Stderr = Join-Path $OutputDir "orchestrator_stderr.log"

$Stages = @(
    @{
        Name = "STAGE 1 - NORMAL VERSUS PNEUMONIA"
        Purpose = "All 4,076 training rows: Normal=0; Viral+Bacterial=Pneumonia=1"
        RunId = "fine_tuning_ft009b_stage1_normal_pneumonia_s42"
    },
    @{
        Name = "STAGE 2 - VIRAL VERSUS BACTERIAL"
        Purpose = "2,971 disease training rows only: Viral=0; Bacterial=1"
        RunId = "fine_tuning_ft009b_stage2_viral_bacterial_s42"
    }
)

function Format-Percent($value) {
    if ($null -eq $value -or "$value" -eq "") { return "Waiting" }
    return ("{0:N3}%" -f ([double]$value * 100.0))
}

function Format-Duration($seconds) {
    if ($null -eq $seconds -or "$seconds" -eq "") { return "Unknown" }
    $span = [TimeSpan]::FromSeconds([double]$seconds)
    if ($span.TotalHours -ge 1) { return ("{0}h {1}m {2}s" -f [int]$span.TotalHours,$span.Minutes,$span.Seconds) }
    return ("{0}m {1}s" -f $span.Minutes,$span.Seconds)
}

function Progress-Bar($fraction) {
    $width = 40
    $fraction = [Math]::Max(0.0, [Math]::Min(1.0, [double]$fraction))
    $filled = [int][Math]::Floor($fraction * $width)
    return "[" + ("#" * $filled) + ("-" * ($width - $filled)) + "] " + ("{0:N1}%" -f ($fraction * 100.0))
}

do {
    Clear-Host
    Write-Host ("=" * 96)
    Write-Host "FT-009B HIERARCHICAL CLASSIFICATION - LIVE DEVELOPMENT MONITOR"
    Write-Host ("=" * 96)
    Write-Host "Architecture:       Two independent DenseNet-201 classifiers"
    Write-Host "Preprocessing:      VIRIDIS pseudo-color -> Global Histogram Equalization (GE) -> CLAHE"
    Write-Host "Resolution:         224 x 224"
    Write-Host "Seed:               42"
    Write-Host "Locked test:        PROHIBITED / NOT ACCESSED"
    Write-Host "Probability rule:   P(N); P(P)*P(V|P); P(P)*P(B|P)"
    Write-Host "Protocol:           $ProtocolPath"
    Write-Host "Updated:            $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))"

    $runningProcess = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*run_ft009b_hierarchical.py*" }
    Write-Host "Orchestrator:       $(if ($runningProcess) { 'RUNNING PID ' + $runningProcess[0].ProcessId } else { 'NOT RUNNING' })"

    foreach ($stage in $Stages) {
        $RunDir = Join-Path $RepoRoot ("model\results\runs\" + $stage.RunId)
        $StatusPath = Join-Path $RunDir "status.json"
        $LivePath = Join-Path $RunDir "live_progress.json"
        $HistoryPath = Join-Path $RunDir "history.csv"
        Write-Host ""
        Write-Host ("-" * 96)
        Write-Host $stage.Name
        Write-Host ("-" * 96)
        Write-Host "Purpose:            $($stage.Purpose)"
        Write-Host "Run ID:             $($stage.RunId)"
        if (-not (Test-Path $StatusPath)) {
            Write-Host "Status:             QUEUED / NOT STARTED"
            continue
        }
        $status = Get-Content -Raw $StatusPath | ConvertFrom-Json
        Write-Host "Status:             $($status.status.ToUpper())"
        Write-Host "Epoch checkpoint:   $($status.current_epoch)/$($status.total_epochs)"
        Write-Host "Best epoch:         $(if ($null -eq $status.best_epoch) { 'Waiting' } else { $status.best_epoch })"
        Write-Host "Best val macro F1:  $(Format-Percent $status.best_validation_macro_f1)"
        Write-Host "Started:            $($status.started_at)"
        Write-Host "Last update:        $($status.updated_at)"

        if (Test-Path $LivePath) {
            $live = Get-Content -Raw $LivePath | ConvertFrom-Json
            Write-Host "Current activity:   $($live.stage)"
            Write-Host "Current epoch:      $($live.epoch)/$($live.total_epochs)"
            if ($null -ne $live.batch) {
                Write-Host "Epoch batches:      $($live.batch)/$($live.total_batches)"
            }
            Write-Host "Epoch progress:     $(Progress-Bar $live.fraction)"
            Write-Host "Running train loss: $(if ($null -eq $live.running_train_loss) { 'Waiting' } else { '{0:N6}' -f [double]$live.running_train_loss })"
            Write-Host "Epoch elapsed:      $(Format-Duration $live.elapsed_seconds)"
            Write-Host "Epoch time left*:   $(Format-Duration $live.estimated_remaining_seconds)"
        }

        if (Test-Path $HistoryPath) {
            $latest = Import-Csv $HistoryPath | Select-Object -Last 1
            Write-Host ""
            Write-Host "Latest completed epoch metrics"
            Write-Host "  Train loss:       $($latest.train_loss)"
            Write-Host "  Validation loss:  $($latest.val_loss)"
            Write-Host "  Macro F1:         $(Format-Percent $latest.val_macro_f1)"
            Write-Host "  Balanced accuracy:$(Format-Percent $latest.val_balanced_accuracy)"
            Write-Host "  Accuracy:         $(Format-Percent $latest.val_accuracy)"
            if ($latest.recall_normal) { Write-Host "  Normal recall:    $(Format-Percent $latest.recall_normal)" }
            if ($latest.recall_pneumonia) { Write-Host "  Pneumonia recall: $(Format-Percent $latest.recall_pneumonia)" }
            if ($latest.recall_viral) { Write-Host "  Viral recall:     $(Format-Percent $latest.recall_viral)" }
            if ($latest.recall_bacterial) { Write-Host "  Bacterial recall: $(Format-Percent $latest.recall_bacterial)" }
            Write-Host "  Learning rate:    $($latest.learning_rate)"
            Write-Host "  Epoch duration:   $(Format-Duration $latest.elapsed_seconds)"
        }
    }

    Write-Host ""
    Write-Host ("-" * 96)
    Write-Host "FINAL THREE-CLASS COMBINATION"
    Write-Host ("-" * 96)
    if (Test-Path $ResultPath) {
        $result = Get-Content -Raw $ResultPath | ConvertFrom-Json
        $metrics = $result.hierarchical_summary.metrics
        Write-Host "Status:             COMPLETED"
        Write-Host "Macro F1:           $(Format-Percent $metrics.macro_f1)"
        Write-Host "Balanced accuracy:  $(Format-Percent $metrics.balanced_accuracy)"
        Write-Host "Accuracy:           $(Format-Percent $metrics.accuracy)"
        Write-Host "Normal recall:      $(Format-Percent $metrics.per_class.Normal.recall)"
        Write-Host "Viral recall:       $(Format-Percent $metrics.per_class.Viral.recall)"
        Write-Host "Bacterial recall:   $(Format-Percent $metrics.per_class.Bacterial.recall)"
        Write-Host "Seed-42 gate:       $(if ($result.seed42_advancement_gate.passed) { 'PASS - MULTI-SEED MAY BE DESIGNED' } else { 'FAIL - STOP AFTER SEED 42' })"
    } else {
        Write-Host "Status:             Waiting for both completed component checkpoints"
    }

    if (Test-Path $Stderr) {
        $errors = Get-Content $Stderr | Select-Object -Last 5
        if ($errors) {
            Write-Host ""
            Write-Host "Latest orchestrator stderr lines"
            $errors | ForEach-Object { Write-Host "  $_" }
        }
    }

    Write-Host ""
    Write-Host "* Time-left values are live estimates for the current epoch, not guaranteed completion times."
    if (-not $Once) { Start-Sleep -Seconds $RefreshSeconds }
} until ($Once)
