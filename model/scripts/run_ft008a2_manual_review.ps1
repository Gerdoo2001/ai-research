$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$host.UI.RawUI.WindowTitle = "FT-008A2 U-Net Manual Anatomical Review"
Clear-Host
Write-Host "FT-008A2 CORRECTED U-NET MANUAL REVIEW" -ForegroundColor Cyan
Write-Host ("=" * 76) -ForegroundColor DarkGray
Write-Host "Each screen shows: original image | Classic U-Net | Legacy U-Net"
Write-Host ""
Write-Host "Press 1  Classic U-Net acceptable; Legacy unacceptable" -ForegroundColor Green
Write-Host "Press 2  Legacy U-Net acceptable; Classic unacceptable" -ForegroundColor Green
Write-Host "Press 3  Both models acceptable" -ForegroundColor Green
Write-Host "Press 0  Neither model acceptable" -ForegroundColor Yellow
Write-Host "Press B  Go to the previous image"
Write-Host "Press Q  Pause safely; completed decisions remain saved" -ForegroundColor Cyan
Write-Host ""
Write-Host "Decision standard: approve only when the mask follows the visible lungs and the"
Write-Host "yellow proposed crop retains both lungs without an anatomically harmful cut."
Write-Host "Automatic PASS/FLAG is supporting information, not the human decision."
Write-Host "Locked test and classifier training remain blocked." -ForegroundColor Yellow
Write-Host ("=" * 76) -ForegroundColor DarkGray

& "venv\Scripts\python.exe" "model\scripts\review_unet_roi_masks_gui.py"
if ($LASTEXITCODE -ne 0) {
    throw "Manual-review GUI exited with code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Review window closed. Saved progress:" -ForegroundColor Cyan
& "venv\Scripts\python.exe" "model\scripts\prepare_unet_roi_manual_review.py"
Write-Host ""
Read-Host "Press Enter to close this terminal"
