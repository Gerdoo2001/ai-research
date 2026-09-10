$ErrorActionPreference = 'SilentlyContinue'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$OutputRoot = Join-Path $RepoRoot 'model\results\ft010_label_smoothing_multiseed'
$RunsRoot = Join-Path $RepoRoot 'model\results\runs'
$RunDefinitions = @(
    @{ Seed = 42; RunId = 'fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42'; Role = 'REFERENCE' },
    @{ Seed = 123; RunId = 'confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s123'; Role = 'CONFIRMATION' },
    @{ Seed = 2026; RunId = 'confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s2026'; Role = 'CONFIRMATION' }
)

function Read-JsonFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $stream = $null
    $reader = $null
    try {
        $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
        $stream = [System.IO.File]::Open($Path, 'Open', 'Read', $share)
        $reader = New-Object System.IO.StreamReader($stream)
        return ($reader.ReadToEnd() | ConvertFrom-Json)
    } catch { return $null }
    finally {
        if ($null -ne $reader) { $reader.Dispose() }
        elseif ($null -ne $stream) { $stream.Dispose() }
    }
}

function Percent($Value) {
    if ($null -eq $Value -or "$Value" -eq '') { return 'Waiting' }
    try { return ('{0:N3}%' -f ([double]$Value * 100.0)) } catch { return 'Waiting' }
}

function Duration($Seconds) {
    if ($null -eq $Seconds -or "$Seconds" -eq '') { return '-' }
    try {
        $span = [TimeSpan]::FromSeconds([Math]::Max(0, [double]$Seconds))
        if ($span.TotalHours -ge 1) { return ('{0}h {1:00}m {2:00}s' -f [int]$span.TotalHours,$span.Minutes,$span.Seconds) }
        return ('{0}m {1:00}s' -f $span.Minutes,$span.Seconds)
    } catch { return '-' }
}

function Progress-Bar($Fraction, [int]$Width = 38) {
    try { $value = [Math]::Max(0.0, [Math]::Min(1.0, [double]$Fraction)) } catch { $value = 0.0 }
    $filled = [int][Math]::Round($value * $Width)
    return '[' + ('#' * $filled) + ('-' * ($Width - $filled)) + ('] {0,5:N1}%' -f ($value * 100.0))
}

function Show-Gpu {
    $line = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($line) {
        $parts = $line -split ',' | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 5) {
            Write-Host ("GPU:                 {0} | use {1}% | memory {2}/{3} MiB | {4} C" -f $parts[0],$parts[1],$parts[2],$parts[3],$parts[4])
            return
        }
    }
    Write-Host 'GPU:                 status unavailable'
}

$Host.UI.RawUI.WindowTitle = 'FT-010A | Label Smoothing Multi-Seed Confirmation'

while ($true) {
    Clear-Host
    $progress = Read-JsonFile (Join-Path $OutputRoot 'progress.json')
    $summary = Read-JsonFile (Join-Path $OutputRoot 'summary.json')

    Write-Host 'FT-010A - DENSENET-201 LABEL-SMOOTHING MULTI-SEED CONFIRMATION' -ForegroundColor Cyan
    Write-Host ('=' * 100)
    Write-Host ("Updated:             {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Show-Gpu
    Write-Host 'Queue status:        ' -NoNewline
    Write-Host $(if ($null -eq $progress) { 'INITIALIZING' } else { ([string]$progress.status).ToUpperInvariant() }) -ForegroundColor Yellow
    Write-Host 'Locked test:         NOT ACCESSED; training and validation only' -ForegroundColor Green
    Write-Host ''
    Write-Host 'FIXED CONFIGURATION'
    Write-Host ('-' * 100)
    Write-Host 'Model:               DenseNet-201 with ImageNet initialization'
    Write-Host 'Pipeline:            VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE'
    Write-Host 'Input / batch:       224 x 224 / 8'
    Write-Host 'Loss:                balanced cross-entropy with label smoothing = 0.05'
    Write-Host 'Optimizer:           AdamW | learning rate 0.0001 | weight decay 0.0001'
    Write-Host 'Scheduler:           ReduceLROnPlateau | factor 0.5 | patience 2'
    Write-Host 'Augmentation:        horizontal flip p=0.5; random rotation +/-5 degrees'
    Write-Host 'Budget:              maximum 12 epochs | early-stop patience 4'
    Write-Host 'Only changed field:  random seed (42, 123, 2026)'
    Write-Host ''

    $current = $null
    if ($null -ne $progress -and $progress.current_run_id) {
        $current = $RunDefinitions | Where-Object { $_.RunId -eq $progress.current_run_id } | Select-Object -First 1
    }
    if ($null -ne $current) {
        $runRoot = Join-Path $RunsRoot $current.RunId
        $status = Read-JsonFile (Join-Path $runRoot 'status.json')
        $live = Read-JsonFile (Join-Path $runRoot 'live_progress.json')
        Write-Host 'CURRENT RUN'
        Write-Host ('-' * 100)
        Write-Host ("Seed:                {0}" -f $current.Seed)
        Write-Host ("Run ID:              {0}" -f $current.RunId)
        Write-Host ("Trainer PID:         {0}" -f $progress.trainer_pid)
        Write-Host ("Training status:     {0}" -f $status.status)
        Write-Host ("Completed epochs:    {0}/{1}" -f $status.current_epoch,$status.total_epochs)
        Write-Host ("Active epoch:        {0}/{1} ({2})" -f $live.epoch,$live.total_epochs,$live.stage)
        Write-Host ("Current epoch:       {0}" -f (Progress-Bar $live.fraction))
        Write-Host ("Current batch:       {0}/{1}" -f $live.batch,$live.total_batches)
        Write-Host ("Epoch elapsed:       {0}" -f (Duration $live.elapsed_seconds))
        Write-Host ("Epoch time left*:    {0}" -f (Duration $live.estimated_remaining_seconds))
        Write-Host ("Best epoch:          {0}" -f $status.best_epoch)
        Write-Host ("Best validation F1:  {0}" -f (Percent $status.best_validation_macro_f1))
        Write-Host ''
    }

    Write-Host 'THREE-SEED TABLE'
    Write-Host ('-' * 100)
    foreach ($definition in $RunDefinitions) {
        $runRoot = Join-Path $RunsRoot $definition.RunId
        $status = Read-JsonFile (Join-Path $runRoot 'status.json')
        $metrics = Read-JsonFile (Join-Path $runRoot 'metrics.json')
        $state = if ($null -eq $status) { 'pending' } else { [string]$status.status }
        Write-Host ("Seed {0,-4} | {1,-12} | {2,-10} | epoch {3,2}/12 | macro F1 {4,-10} | accuracy {5}" -f $definition.Seed,$definition.Role,$state,$status.current_epoch,(Percent $metrics.macro_f1),(Percent $metrics.accuracy))
    }
    if ($null -ne $summary) {
        Write-Host ''
        Write-Host ("Completed seeds:     {0}/{1}" -f $summary.completed_seed_count,$summary.target_seed_count)
        Write-Host ("Current mean F1:     {0}" -f (Percent $summary.mean_macro_f1))
        Write-Host ("Sample std. F1:      {0}" -f (Percent $summary.sample_std_macro_f1))
    }
    Write-Host ''
    Write-Host '*Time estimates stabilize after the first epoch.'
    Write-Host 'Closing this monitor does not stop the hidden training queue.' -ForegroundColor DarkGray
    if ($null -ne $progress -and $progress.status -in @('completed','failed')) {
        Write-Host 'Queue has reached a terminal state; this screen will remain available.' -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 3
}
