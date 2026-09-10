$ErrorActionPreference = 'SilentlyContinue'
$Host.UI.RawUI.WindowTitle = 'FT-003 DenseNet-201 Mild Affine - Live Training Monitor'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RunId = 'tuning_densenet201_viridis_ge_clahe_mild_affine_s42'
$RunRoot = Join-Path $RepoRoot "model\results\runs\$RunId"

function Read-SharedJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            $share
        )
        try {
            $reader = [System.IO.StreamReader]::new($stream)
            try {
                $content = $reader.ReadToEnd()
                if ($content) { return ($content | ConvertFrom-Json) }
            }
            finally { $reader.Dispose() }
        }
        finally { $stream.Dispose() }
    }
    catch { return $null }
}

function Percent([object]$Value) {
    if ($null -eq $Value -or $Value -eq '') { return '-' }
    return ('{0:N3}%' -f (100.0 * [double]$Value))
}

function Duration([object]$Seconds) {
    if ($null -eq $Seconds -or $Seconds -eq '') { return '-' }
    $span = [TimeSpan]::FromSeconds([double]$Seconds)
    if ($span.TotalHours -ge 1) { return ('{0}h {1}m {2}s' -f [int]$span.TotalHours, $span.Minutes, $span.Seconds) }
    return ('{0}m {1}s' -f $span.Minutes, $span.Seconds)
}

function ProgressBar([double]$Fraction, [int]$Width = 42) {
    $bounded = [Math]::Max(0.0, [Math]::Min(1.0, $Fraction))
    $filled = [int][Math]::Floor($bounded * $Width)
    return ('[' + ('#' * $filled) + ('-' * ($Width - $filled)) + ']')
}

while ($true) {
    $status = Read-SharedJson (Join-Path $RunRoot 'status.json')
    $live = Read-SharedJson (Join-Path $RunRoot 'live_progress.json')
    $metrics = Read-SharedJson (Join-Path $RunRoot 'metrics.json')
    $history = $null
    $historyPath = Join-Path $RunRoot 'history.csv'
    if (Test-Path -LiteralPath $historyPath) {
        try { $history = @(Import-Csv -LiteralPath $historyPath | Select-Object -Last 1)[0] } catch {}
    }

    Clear-Host
    Write-Host ('=' * 92) -ForegroundColor Cyan
    Write-Host 'FT-003 LIVE TRAINING MONITOR' -ForegroundColor Cyan
    Write-Host ('=' * 92) -ForegroundColor Cyan
    Write-Host 'Model:                 DenseNet-201'
    Write-Host 'Preprocessing:         VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE'
    Write-Host 'Training augmentation: horizontal flip 50%; rotation +/-5 deg; translation +/-3%; scale 0.95-1.05'
    Write-Host 'Input / batch / seed:  224 x 224 / 8 / 42'
    Write-Host "Run ID:                $RunId"
    Write-Host 'Locked test accessed:  No'
    Write-Host ''

    if ($null -eq $status) {
        Write-Host 'Waiting for the trainer status file...' -ForegroundColor Yellow
    }
    else {
        $state = [string]$status.status
        $stateColor = if ($state -eq 'completed') { 'Green' } elseif ($state -eq 'failed') { 'Red' } else { 'Yellow' }
        Write-Host ("Training status:       {0}" -f $state.ToUpperInvariant()) -ForegroundColor $stateColor
        Write-Host ("Completed epochs:      {0}/{1}" -f $status.current_epoch, $status.total_epochs)
        Write-Host ("Best epoch:            {0}" -f $(if ($status.best_epoch) { $status.best_epoch } else { '-' }))
        Write-Host ("Best validation F1:    {0}" -f (Percent $status.best_validation_macro_f1))

        if ($state -eq 'running' -and $null -ne $live) {
            $fraction = if ($null -ne $live.fraction) { [double]$live.fraction } else { 0.0 }
            Write-Host ''
            Write-Host 'CURRENT EPOCH' -ForegroundColor Cyan
            Write-Host ('-' * 92)
            Write-Host ("Epoch:                  {0}/{1}" -f $live.epoch, $live.total_epochs)
            Write-Host ("Batch:                  {0}/{1}" -f $live.batch, $live.total_batches)
            Write-Host ("Progress:               {0} {1:N1}%" -f (ProgressBar $fraction), (100.0 * $fraction)) -ForegroundColor Green
            Write-Host ("Running train loss:     {0:N4}" -f [double]$live.running_train_loss)
            Write-Host ("Epoch elapsed:          {0}" -f (Duration $live.elapsed_seconds))
            Write-Host ("Estimated epoch time:   {0}" -f (Duration $live.estimated_remaining_seconds))
        }

        if ($null -ne $history) {
            Write-Host ''
            Write-Host 'LATEST COMPLETED EPOCH' -ForegroundColor Cyan
            Write-Host ('-' * 92)
            Write-Host ("Epoch:                  {0}" -f $history.epoch)
            Write-Host ("Validation macro F1:    {0}" -f (Percent $history.val_macro_f1))
            Write-Host ("Balanced accuracy:      {0}" -f (Percent $history.val_balanced_accuracy))
            Write-Host ("Accuracy:               {0}" -f (Percent $history.val_accuracy))
            Write-Host ("Viral recall:           {0}" -f (Percent $history.recall_viral))
        }

        if ($state -eq 'completed' -and $null -ne $metrics) {
            Write-Host ''
            Write-Host 'FINAL VALIDATION RESULT' -ForegroundColor Green
            Write-Host ('-' * 92)
            Write-Host ("Macro F1:               {0}" -f (Percent $metrics.macro_f1))
            Write-Host ("Balanced accuracy:      {0}" -f (Percent $metrics.balanced_accuracy))
            Write-Host ("Accuracy:               {0}" -f (Percent $metrics.accuracy))
            Write-Host ("Viral recall:           {0}" -f (Percent $metrics.per_class.Viral.recall))
            Write-Host ("Training duration:      {0}" -f (Duration $metrics.training_duration_seconds))
        }

        if ($state -eq 'failed') {
            Write-Host ("Failure reason:         {0}" -f $status.failure_reason) -ForegroundColor Red
        }
    }

    Write-Host ''
    Write-Host ('Last refreshed: {0}    Refresh: 2 seconds    Close this window to stop monitoring only.' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
}
