# start_gpu_training.ps1
# Script to correctly set up and run GPU training for VA Scanner

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   VA SCANNER - GPU TRAINING LAUNCHER" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Define paths
$ProjectRoot = "C:\Users\user\Desktop\Project Final University Bon\sp-project-academy-va-scan"
$VenvPath = "$ProjectRoot\venv_gpu"
$PythonPath = "$VenvPath\Scripts\python.exe"
$TrainScript = "$ProjectRoot\backend\training\train_enhanced.py"
$InspectScript = "$ProjectRoot\backend\inspect_data.py"

# 2. Check Virtual Environment
if (-not (Test-Path $PythonPath)) {
    Write-Host "❌ Error: venv_gpu not found at $VenvPath" -ForegroundColor Red
    Write-Host "   Please create it using: python -m venv venv_gpu"
    Exit 1
}
Write-Host "✅ Virtual Environment found." -ForegroundColor Green

# 3. Check GPU Availability (using the venv python)
Write-Host "[1/3] Checking GPU Status..." -ForegroundColor Yellow
$GpuCheckCode = "import torch; print(f'GPU Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'} | CUDA: {torch.version.cuda}')"
& $PythonPath -c $GpuCheckCode

# 4. Check Data
Write-Host "`n[2/3] Verifying Dataset..." -ForegroundColor Yellow
if (Test-Path $InspectScript) {
    & $PythonPath $InspectScript
} else {
    Write-Host "⚠️ Warning: correct inspect_data.py not found, skipping." -ForegroundColor Yellow
}

# 5. Start Training
Write-Host "`n[3/3] Starting Training..." -ForegroundColor Yellow
Write-Host "   Script: $TrainScript"
# Change to backend directory so imports work correctly
Set-Location "$ProjectRoot\backend"
Write-Host "   Working Directory: $(Get-Location)"

# Execute training
try {
    & $PythonPath "training/train_enhanced.py"
} catch {
    Write-Host "`n❌ Training failed with error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host "`nDone." -ForegroundColor Cyan
