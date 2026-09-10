$ErrorActionPreference = 'SilentlyContinue'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ClaheRoot = Join-Path $ProjectRoot 'model\results\densenet201_clahe_multiseed'
$TuningRoot = Join-Path $ProjectRoot 'model\results\densenet201_320_tuning'
$RunsRoot = Join-Path $ProjectRoot 'model\results\runs'
$TuningRunId = 'tuning_densenet201_viridis_ge_clahe_320_b8_s42'
$TuningRunRoot = Join-Path $RunsRoot $TuningRunId

function Read-JsonFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $stream = $null
    $reader = $null
    try {
        # FileShare.Delete is required because the Python trainers publish JSON
        # atomically by replacing a temporary file. A normal Get-Content can
        # briefly block that replacement on Windows.
        $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            $share
        )
        $reader = New-Object System.IO.StreamReader($stream)
        $text = $reader.ReadToEnd()
        return $text | ConvertFrom-Json
    } catch {
        return $null
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
        elseif ($null -ne $stream) { $stream.Dispose() }
    }
}

function Percent($Value) {
    if ($null -eq $Value -or "$Value" -eq '') { return 'Waiting' }
    try { return ('{0:N3}%' -f ([double]$Value * 100.0)) } catch { return 'Waiting' }
}

function Progress-Bar($Fraction, [int]$Width = 36) {
    try { $value = [Math]::Max(0.0, [Math]::Min(1.0, [double]$Fraction)) } catch { $value = 0.0 }
    $filled = [int][Math]::Round($value * $Width)
    return '[' + ('#' * $filled) + ('-' * ($Width - $filled)) + ('] {0,5:N1}%' -f ($value * 100.0))
}

function Duration($Seconds) {
    if ($null -eq $Seconds -or "$Seconds" -eq '') { return '-' }
    try {
        $span = [TimeSpan]::FromSeconds([Math]::Max(0, [double]$Seconds))
        if ($span.TotalHours -ge 1) { return ('{0}h {1:00}m {2:00}s' -f [int]$span.TotalHours, $span.Minutes, $span.Seconds) }
        return ('{0}m {1:00}s' -f $span.Minutes, $span.Seconds)
    } catch { return '-' }
}

function Show-Gpu {
    $line = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($line) {
        $parts = $line -split ',' | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 5) {
            Write-Host ("GPU: {0} | use {1}% | memory {2}/{3} MiB | {4} C" -f $parts[0],$parts[1],$parts[2],$parts[3],$parts[4])
            return
        }
    }
    Write-Host 'GPU: status unavailable'
}

$Host.UI.RawUI.WindowTitle = 'Pneumonia Training Queue - Live Monitor'

while ($true) {
    Clear-Host
    Write-Host 'PNEUMONIA MODEL TRAINING - LIVE QUEUE MONITOR'
    Write-Host ('=' * 94)
    Write-Host ("Updated: {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Show-Gpu
    Write-Host 'Locked test: NOT ACCESSED (training and validation only)'
    Write-Host ''

    $clahe = Read-JsonFile (Join-Path $ClaheRoot 'progress.json')
    $claheSummary = Read-JsonFile (Join-Path $ClaheRoot 'summary.json')
    Write-Host 'QUEUE 1 - DENSENET-201 + CLAHE ALONE, THREE-SEED CONFIRMATION'
    Write-Host ('-' * 94)
    if ($null -eq $clahe) {
        Write-Host 'Status: progress file not available'
    } else {
        Write-Host ("Queue status:         {0}" -f $clahe.status)
        Write-Host ("Current seed:         {0}" -f $clahe.current_seed)
        Write-Host ("Current run ID:       {0}" -f $clahe.current_run_id)
        $trainStatus = $clahe.training_status
        $live = $clahe.live_progress
        Write-Host ("Training status:      {0}" -f $trainStatus.status)
        Write-Host ("Completed epochs:     {0}/{1}" -f $trainStatus.current_epoch,$trainStatus.total_epochs)
        Write-Host ("Active epoch:         {0}/{1} ({2})" -f $live.epoch,$live.total_epochs,$live.stage)
        Write-Host ("Current epoch:        {0}" -f (Progress-Bar $live.fraction))
        Write-Host ("Current batch:        {0}/{1}" -f $live.batch,$live.total_batches)
        Write-Host ("Epoch time left*:     {0}" -f (Duration $live.estimated_remaining_seconds))
        Write-Host ("Best validation F1:   {0}" -f (Percent $trainStatus.best_validation_macro_f1))
        if ($null -ne $claheSummary -and $null -ne $claheSummary.runs) {
            Write-Host ''
            Write-Host 'Seed status:'
            foreach ($run in $claheSummary.runs) {
                Write-Host ("  Seed {0,-4} | {1,-10} | best macro F1 {2}" -f $run.seed,$run.status,(Percent $run.macro_f1))
            }
        }
    }
    Write-Host ''

    $tuning = Read-JsonFile (Join-Path $TuningRoot 'progress.json')
    $tuningStatus = Read-JsonFile (Join-Path $TuningRunRoot 'status.json')
    $tuningLive = Read-JsonFile (Join-Path $TuningRunRoot 'live_progress.json')
    $tuningMetrics = Read-JsonFile (Join-Path $TuningRunRoot 'metrics.json')
    Write-Host 'QUEUE 2 - DENSENET-201 + VIRIDIS -> GLOBAL HISTOGRAM EQUALIZATION (GE) -> CLAHE'
    Write-Host '          320 x 320 RESOLUTION FINE-TUNING EXPERIMENT'
    Write-Host ('-' * 94)
    if ($null -eq $tuning -and $null -eq $tuningStatus) {
        Write-Host 'Status: progress file not available'
    } else {
        Write-Host 'Input resolution:     320 x 320'
        Write-Host 'Seed:                 42'
        Write-Host 'Selected batch size:  8 (CUDA benchmark passed)'
        Write-Host ("Run ID:               {0}" -f $TuningRunId)
        if ($null -ne $tuningStatus) {
            Write-Host ("Actual trainer:       {0}" -f $tuningStatus.status)
            if ($null -ne $tuning -and $tuning.status -eq 'failed' -and $tuningStatus.status -eq 'running') {
                Write-Host 'Queue controller:     detached after a progress-file sharing error; trainer is healthy'
            } elseif ($null -ne $tuning) {
                Write-Host ("Queue controller:     {0}" -f $tuning.status)
            }
            Write-Host ("Completed epochs:     {0}/{1}" -f $tuningStatus.current_epoch,$tuningStatus.total_epochs)
            Write-Host ("Best epoch:           {0}" -f $tuningStatus.best_epoch)
            Write-Host ("Best validation F1:   {0}" -f (Percent $tuningStatus.best_validation_macro_f1))
        } elseif ($null -ne $tuning) {
            Write-Host ("Queue status:         {0}" -f $tuning.status)
            if ($tuning.blocking_run_id) { Write-Host ("Waiting behind:       {0}" -f $tuning.blocking_run_id) }
        }
        if ($null -ne $tuningLive -and $tuningStatus.status -eq 'running') {
            Write-Host ("Active epoch:         {0}/{1} ({2})" -f $tuningLive.epoch,$tuningLive.total_epochs,$tuningLive.stage)
            Write-Host ("Current epoch:        {0}" -f (Progress-Bar $tuningLive.fraction))
            Write-Host ("Current batch:        {0}/{1}" -f $tuningLive.batch,$tuningLive.total_batches)
            Write-Host ("Epoch elapsed:        {0}" -f (Duration $tuningLive.elapsed_seconds))
            Write-Host ("Epoch time left*:     {0}" -f (Duration $tuningLive.estimated_remaining_seconds))
        }
        if ($null -ne $tuningMetrics -and $tuningStatus.status -eq 'completed') {
            Write-Host ''
            Write-Host 'Final validation metrics:'
            Write-Host ("  Macro F1:           {0}" -f (Percent $tuningMetrics.macro_f1))
            Write-Host ("  Balanced accuracy:  {0}" -f (Percent $tuningMetrics.balanced_accuracy))
            Write-Host ("  Accuracy:           {0}" -f (Percent $tuningMetrics.accuracy))
            Write-Host ("  Viral recall:       {0}" -f (Percent $tuningMetrics.per_class.Viral.recall))
        }
    }
    Write-Host ''
    Write-Host '*Time estimates stabilize after the first epoch.'
    Write-Host 'This window refreshes every 3 seconds. Close it to stop monitoring only; training continues.'
    Start-Sleep -Seconds 3
}
