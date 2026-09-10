$ErrorActionPreference = 'SilentlyContinue'
$Host.UI.RawUI.WindowTitle = 'Pneumonia Improvement Queue - ALL Live Details'
try {
    $Host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size(190, 500)
    $Host.UI.RawUI.WindowSize = New-Object Management.Automation.Host.Size(190, 62)
} catch {}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ProgressPath = Join-Path $RepoRoot 'model\results\improvement_queue\progress.json'

function Read-SharedJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
        try {
            $reader = [System.IO.StreamReader]::new($stream)
            try {
                $text = $reader.ReadToEnd()
                if ($text) { return ($text | ConvertFrom-Json) }
            } finally { $reader.Dispose() }
        } finally { $stream.Dispose() }
    } catch { return $null }
}

function Pct([object]$Value, [int]$Digits = 3) {
    if ($null -eq $Value -or $Value -eq '') { return '-' }
    return (('{0:N' + $Digits + '}%') -f (100.0 * [double]$Value))
}

function Dur([object]$Seconds) {
    if ($null -eq $Seconds -or $Seconds -eq '') { return '-' }
    $span = [TimeSpan]::FromSeconds([Math]::Max(0, [double]$Seconds))
    if ($span.TotalHours -ge 1) { return ('{0}h {1:D2}m {2:D2}s' -f [int]$span.TotalHours, $span.Minutes, $span.Seconds) }
    return ('{0}m {1:D2}s' -f $span.Minutes, $span.Seconds)
}

function Bar([double]$Fraction, [int]$Width = 46) {
    $bounded = [Math]::Max(0, [Math]::Min(1, $Fraction))
    $filled = [int][Math]::Floor($bounded * $Width)
    return ('[' + ('#' * $filled) + ('-' * ($Width - $filled)) + ']')
}

function Line([string]$Label, [object]$Value, [string]$Color = 'Gray') {
    Write-Host ("{0,-31} {1}" -f ($Label + ':'), $Value) -ForegroundColor $Color
}

function Section([string]$Title, [string]$Color = 'Cyan') {
    Write-Host ''
    Write-Host $Title -ForegroundColor $Color
    Write-Host ('-' * 188) -ForegroundColor DarkGray
}

while ($true) {
    $state = Read-SharedJson $ProgressPath
    Clear-Host
    Write-Host ('=' * 188) -ForegroundColor Cyan
    Write-Host 'PNEUMONIA IMPROVEMENT QUEUE - COMPLETE LIVE DETAILS' -ForegroundColor Cyan
    Write-Host 'Stage 1: three-seed ensemble confirmation    Stage 2: TorchXRayVision PSPNet pediatric lung-mask audit' -ForegroundColor Cyan
    Write-Host ('=' * 188) -ForegroundColor Cyan

    if ($null -eq $state) {
        Write-Host 'Waiting for the background controller to create progress.json ...' -ForegroundColor Yellow
        Start-Sleep -Seconds 2
        continue
    }

    $statusColor = if ($state.status -eq 'failed') { 'Red' } elseif ($state.status -match 'awaiting|completed') { 'Green' } else { 'Yellow' }
    Line 'Queue ID' $state.queue_id
    Line 'Overall queue status' ([string]$state.status).ToUpperInvariant() $statusColor
    Line 'Queue started' $state.started_at
    Line 'Last state update' $state.updated_at
    Line 'Controller PID' $state.queue_summary.controller_pid
    Line 'Locked test accessed' 'NO - PROHIBITED DURING BOTH STAGES' 'Green'
    Line 'Completed new training runs' ("{0}/{1}" -f $state.queue_summary.completed_training_runs, $state.queue_summary.total_training_runs)

    Section 'REPRODUCIBILITY AND COMPUTE ENVIRONMENT'
    Line 'Manifest version' $state.scientific_controls.manifest_version
    Line 'Manifest SHA-256' $state.scientific_controls.manifest_hash
    Line 'Split counts' ("train={0}; validation={1}; locked test={2}" -f $state.scientific_controls.train_images, $state.scientific_controls.validation_images, $state.scientific_controls.test_images)
    Line 'Class order' ($state.scientific_controls.class_order -join ' -> ')
    Line 'Checkpoint selection metric' $state.scientific_controls.selection_metric
    Line 'GPU concurrency' 'Exactly one GPU job at a time'
    Line 'Python / executable' ("{0} / {1}" -f $state.environment.python, $state.environment.executable)
    Line 'PyTorch / torchvision' ("{0} / {1}" -f $state.environment.torch, $state.environment.torchvision)
    Line 'NumPy / TorchXRayVision' ("{0} / {1}" -f $state.environment.numpy, $state.environment.torchxrayvision)
    Line 'CUDA / GPU' ("CUDA {0}; {1}" -f $state.environment.cuda_version, $state.environment.gpu)

    $ens = $state.stages[0]
    Section 'STAGE 1 - FT-004 THREE-SEED EQUAL-PROBABILITY ENSEMBLE CONFIRMATION'
    Line 'Stage status' ([string]$ens.status).ToUpperInvariant()
    Line 'Scientific objective' $ens.objective
    Line 'Ensemble rule' $ens.method
    Line 'Frozen member 1' 'DenseNet-201 + VIRIDIS -> GE -> CLAHE'
    Line 'Frozen member 2' 'DenseNet-201 + VIRIDIS -> GE'
    Line 'Frozen member 3' 'DenseNet-201 + GE'
    Line 'Seeds' ($ens.seeds -join ', ')
    Line 'Confirmation reporting' $ens.confirmation_rule
    Line 'Existing seed-42 screening' 'Macro F1 86.953%; accuracy 87.757%; balanced accuracy 86.593%; Viral recall 70.404%'
    Line 'New runs required' ("{0}; completed {1}" -f $ens.total_training_runs, $ens.completed_training_runs)

    Write-Host ''
    Write-Host ('{0,-15} {1,-15} {2,-16} {3,-10} {4,-8} {5,-9} {6,-9} {7,-9} {8,-10}' -f 'ITEM', 'STATUS', 'PREPROCESSING', 'SEED', 'EPOCH', 'MACRO F1', 'ACCURACY', 'VIRAL REC', 'DURATION') -ForegroundColor White
    foreach ($item in $ens.training_runs) {
        $snap = $item.snapshot
        $runStatus = if ($item.status) { $item.status } else { 'pending' }
        $epoch = if ($snap.status.current_epoch) { "{0}/{1}" -f $snap.status.current_epoch, $snap.status.total_epochs } else { '-' }
        $metrics = $snap.metrics
        $duration = if ($metrics.training_duration_seconds) { Dur $metrics.training_duration_seconds } else { '-' }
        Write-Host ('{0,-15} {1,-15} {2,-16} {3,-10} {4,-8} {5,-9} {6,-9} {7,-9} {8,-10}' -f $item.item_id, $runStatus, $item.member, $item.seed, $epoch, (Pct $metrics.macro_f1), (Pct $metrics.accuracy), (Pct $metrics.per_class.Viral.recall), $duration)
    }

    Section 'ALL TRAINING CONFIGURATION DETAILS FOR THE FOUR QUEUED RUNS'
    foreach ($item in $ens.training_runs) {
        $cfg = $item.configuration
        Write-Host ("{0} | {1} | run={2}" -f $item.item_id, ([string]$item.status).ToUpperInvariant(), $item.run_id) -ForegroundColor Yellow
        Write-Host ("  model={0}; weights={1}; preprocessing={2}; seed={3}; input={4}x{4}; batch={5}; epochs={6}; CUDA={7}; AMP={8}" -f $cfg.architecture, $cfg.weights, $cfg.preprocessing, $cfg.seed, $cfg.input_size, $cfg.batch_size, $cfg.epochs, $cfg.device, $cfg.amp)
        Write-Host ("  optimizer={0}; LR={1}; weight_decay={2}; scheduler={3}; scheduler_patience={4}; scheduler_factor={5}; early_stop_patience={6}; minimum_delta={7}" -f $cfg.optimizer, $cfg.learning_rate, $cfg.weight_decay, $cfg.scheduler, $cfg.scheduler_patience, $cfg.scheduler_factor, $cfg.early_stopping_patience, $cfg.minimum_delta)
        Write-Host ("  loss={0}; class_weighting={1}; label_smoothing={2}; freeze_backbone={3}; augmentation=horizontal flip p={4}, rotation +/-{5} degrees" -f $cfg.loss, $cfg.class_weighting, $cfg.label_smoothing, $cfg.freeze_backbone, $cfg.augmentation.horizontal_flip_probability, $cfg.augmentation.rotation_degrees)
        Write-Host ("  config={0}" -f $item.config_path_absolute) -ForegroundColor DarkGray
        Write-Host ("  artifacts={0}" -f $item.snapshot.run_dir) -ForegroundColor DarkGray
    }

    if ($state.current.type -eq 'training') {
        $cur = $state.current
        $snap = $cur.snapshot
        $live = $snap.live_progress
        $latest = $snap.latest_completed_epoch
        Section 'CURRENT TRAINING RUN - LIVE BATCH DETAIL' 'Green'
        Line 'Item / action' ("{0} / {1}" -f $cur.item_id, $cur.action)
        Line 'Run ID' $cur.run_id
        Line 'Process ID' $cur.process_id
        Line 'Run elapsed' (Dur $cur.elapsed_seconds)
        Line 'Trainer status' $snap.status.status
        Line 'Completed epochs' ("{0}/{1}" -f $snap.status.current_epoch, $snap.status.total_epochs)
        Line 'Best epoch / validation F1' ("{0} / {1}" -f $snap.status.best_epoch, (Pct $snap.status.best_validation_macro_f1))
        if ($null -ne $live -and $live.stage -eq 'training') {
            $fraction = if ($null -ne $live.fraction) { [double]$live.fraction } else { 0.0 }
            Line 'Current epoch' ("{0}/{1}" -f $live.epoch, $live.total_epochs)
            Line 'Current batch' ("{0}/{1}" -f $live.batch, $live.total_batches)
            Line 'Current epoch progress' ("{0} {1:N1}%" -f (Bar $fraction), (100.0 * $fraction)) 'Green'
            Line 'Running training loss' ('{0:N6}' -f [double]$live.running_train_loss)
            Line 'Current epoch elapsed' (Dur $live.elapsed_seconds)
            Line 'Current epoch ETA' (Dur $live.estimated_remaining_seconds)
        }
        if ($null -ne $latest) {
            Line 'Latest completed epoch' $latest.epoch
            Line 'Train / validation loss' ("{0} / {1}" -f $latest.train_loss, $latest.val_loss)
            Line 'Validation Macro F1' (Pct $latest.val_macro_f1)
            Line 'Validation balanced acc.' (Pct $latest.val_balanced_accuracy)
            Line 'Validation accuracy' (Pct $latest.val_accuracy)
            Line 'Normal / Viral / Bacterial recall' ("{0} / {1} / {2}" -f (Pct $latest.recall_normal), (Pct $latest.recall_viral), (Pct $latest.recall_bacterial))
            Line 'Learning rate' $latest.learning_rate
            Line 'Epoch duration' (Dur $latest.elapsed_seconds)
        }
        Line 'Best checkpoint' $snap.artifacts.best_checkpoint
        Line 'Last checkpoint' $snap.artifacts.last_checkpoint
        Line 'Stdout / stderr' ("{0} / {1}" -f ($ens.training_runs | Where-Object {$_.run_id -eq $cur.run_id}).stdout_log, ($ens.training_runs | Where-Object {$_.run_id -eq $cur.run_id}).stderr_log)
    }

    if ($ens.result.aggregate) {
        Section 'STAGE 1 FINAL MULTI-SEED ENSEMBLE RESULT' 'Green'
        foreach ($seedResult in $ens.result.per_seed) {
            $m = $seedResult.metrics
            Line ("Seed {0}" -f $seedResult.seed) ("Macro F1={0}; accuracy={1}; balanced={2}; Viral recall={3}" -f (Pct $m.macro_f1), (Pct $m.accuracy), (Pct $m.balanced_accuracy), (Pct $m.per_class.Viral.recall))
        }
        Line 'Mean Macro F1 +/- sample SD' ("{0} +/- {1}" -f (Pct $ens.result.aggregate.macro_f1.mean), (Pct $ens.result.aggregate.macro_f1.sample_std))
        Line 'Mean accuracy +/- sample SD' ("{0} +/- {1}" -f (Pct $ens.result.aggregate.accuracy.mean), (Pct $ens.result.aggregate.accuracy.sample_std))
        Line 'Mean Viral recall +/- SD' ("{0} +/- {1}" -f (Pct $ens.result.aggregate.viral_recall.mean), (Pct $ens.result.aggregate.viral_recall.sample_std))
        Line 'Ensemble summary artifact' $ens.evaluation_output
    }

    $roi = $state.stages[1]
    Section 'STAGE 2 - FT-002 TORCHXRAYVISION PSPNET PEDIATRIC LUNG-ROI AUDIT'
    Line 'Stage status' ([string]$roi.status).ToUpperInvariant()
    Line 'Scientific objective' $roi.objective
    Line 'Package / architecture' ("{0}; {1}" -f $roi.model.package, $roi.model.architecture)
    Line 'Model parameter count' ('{0:N0}' -f [double]$roi.model.parameters)
    Line 'Native input' $roi.model.native_input
    Line 'Input normalization' $roi.model.input_normalization
    Line 'Model output' $roi.model.output
    Line 'Selected output channels' ($roi.model.lung_channels -join ' + ')
    Line 'Mask threshold / crop margin' ("{0} / {1}" -f $roi.model.threshold, $roi.model.crop_margin)
    Line 'Audit source' $roi.audit.source
    Line 'Audit sample' ("{0} per class x 3 = {1}; selection seed {2}" -f $roi.audit.images_per_class, $roi.audit.total_images, $roi.audit.selection_seed)
    Line 'Automated checks' ($roi.audit.automated_checks -join '; ')
    Line 'Automated acceptance gate' $roi.audit.automated_gate
    Line 'Mandatory human gate' $roi.audit.manual_gate 'Yellow'
    Line 'Audit progress artifact' $roi.audit.progress_path
    Line 'Audit final summary' $roi.audit.summary_path

    $roiProgress = if ($state.current.roi_progress) { $state.current.roi_progress } else { $roi.status_detail }
    if ($roiProgress.completed_images -ne $null) {
        $roiFraction = if ([double]$roiProgress.total_images -gt 0) { [double]$roiProgress.completed_images / [double]$roiProgress.total_images } else { 0.0 }
        Section 'CURRENT ROI AUDIT - LIVE IMAGE DETAIL' 'Green'
        Line 'ROI audit state' $roiProgress.status
        Line 'Completed masks' ("{0}/{1}" -f $roiProgress.completed_images, $roiProgress.total_images)
        Line 'Audit progress' ("{0} {1:N1}%" -f (Bar $roiFraction), (100.0 * $roiFraction)) 'Green'
        Line 'Current class / image' ("{0} / {1}" -f $roiProgress.current_class, $roiProgress.current_image)
        Line 'Automated passed / flagged' ("{0} / {1}" -f $roiProgress.passed_images, $roiProgress.flagged_images)
        Line 'Elapsed / ETA' ("{0} / {1}" -f (Dur $roiProgress.elapsed_seconds), (Dur $roiProgress.estimated_remaining_seconds))
        if ($roiProgress.current_result) {
            Line 'Current mask left/right/union' ("{0} / {1} / {2}" -f (Pct $roiProgress.current_result.left_fraction), (Pct $roiProgress.current_result.right_fraction), (Pct $roiProgress.current_result.union_fraction))
            Line 'Current raw components L/R' ("{0}/{1}" -f $roiProgress.current_result.left_raw_components, $roiProgress.current_result.right_raw_components)
            Line 'Current automated pass' $roiProgress.current_result.automated_pass
            Line 'Current quality flags' ($roiProgress.current_result.quality_flags -join ', ')
            Line 'Current overlay' $roiProgress.current_result.overlay_path
        }
    }

    if ($roi.result) {
        Section 'ROI AUDIT RESULT AND APPROVAL GATE' 'Yellow'
        Line 'Automated pass rate' (Pct $roi.result.automated_pass_rate)
        Line 'Passed / flagged' ("{0}/{1}" -f $roi.result.automated_pass_count, $roi.result.automated_flagged_count)
        Line 'Automated gate passed' $roi.result.automated_gate_passed
        Line 'Manual review required' $roi.result.manual_review_required
        Line 'Manual review approved' $roi.result.manual_review_approved
        Line 'Contact sheets' ($roi.result.contact_sheets -join '; ')
        Line 'Interpretation boundary' $roi.result.interpretation
    }

    Section 'LATEST QUEUE EVENTS'
    foreach ($entry in @($state.events | Select-Object -Last 12)) {
        $color = if ($entry.level -eq 'ERROR') { 'Red' } elseif ($entry.level -eq 'ACTION_REQUIRED') { 'Yellow' } else { 'DarkGray' }
        Write-Host ("{0} [{1}] {2}" -f $entry.timestamp, $entry.level, $entry.message) -ForegroundColor $color
    }

    Write-Host ''
    Write-Host ('=' * 188) -ForegroundColor Cyan
    Write-Host ("Refresh: every 2 seconds | Screen refreshed: {0} | Closing this window stops monitoring only, not the background queue." -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor DarkGray
    Write-Host 'Power loss, shutdown, or restart stops the active process. Completed epochs are checkpointed; rerunning the controller resumes/skips safely.' -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
}
