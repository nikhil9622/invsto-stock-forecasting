# Project Verification Script
# Checks if all files and dependencies are in place

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project Verification Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Check Python version
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -ge 3 -and $minor -ge 8) {
            Write-Host "  ✓ Python $pythonVersion found" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Python 3.8+ required (found $pythonVersion)" -ForegroundColor Red
            $errors++
        }
    }
} catch {
    Write-Host "  ✗ Python not found in PATH" -ForegroundColor Red
    $errors++
}

# Check directory structure
Write-Host "`n[2/5] Checking directory structure..." -ForegroundColor Yellow
$requiredDirs = @(
    "notebooks",
    "src",
    "data/raw",
    "data/processed",
    "models",
    "report"
)

foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "  ✓ $dir/" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dir/ missing" -ForegroundColor Red
        $errors++
    }
}

# Check source files
Write-Host "`n[3/5] Checking source files..." -ForegroundColor Yellow
$requiredFiles = @(
    "src/data_download.py",
    "src/data_cleaning.py",
    "src/features.py",
    "src/train_arima.py",
    "src/train_xgb.py",
    "src/backtest.py",
    "notebooks/01_stock_forecasting_eda.ipynb",
    "requirements.txt",
    "README.md"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file missing" -ForegroundColor Red
        $errors++
    }
}

# Check Python packages
Write-Host "`n[4/5] Checking Python packages..." -ForegroundColor Yellow
$requiredPackages = @(
    "yfinance",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "mplfinance",
    "statsmodels",
    "pmdarima",
    "scikit-learn",
    "xgboost",
    "ta",
    "joblib"
)

$installedPackages = pip list 2>&1

foreach ($package in $requiredPackages) {
    if ($installedPackages -match $package) {
        Write-Host "  ✓ $package" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $package not installed" -ForegroundColor Yellow
        $warnings++
    }
}

# Check for data
Write-Host "`n[5/5] Checking for downloaded data..." -ForegroundColor Yellow
$csvFiles = Get-ChildItem -Path "data/raw" -Filter "*.csv" -ErrorAction SilentlyContinue
if ($csvFiles.Count -gt 0) {
    Write-Host "  ✓ Found $($csvFiles.Count) CSV file(s) in data/raw/" -ForegroundColor Green
    foreach ($file in $csvFiles) {
        Write-Host "    - $($file.Name)" -ForegroundColor Gray
    }
} else {
    Write-Host "  ⚠ No data files found (run data download first)" -ForegroundColor Yellow
    $warnings++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "Status: All checks passed! ✓" -ForegroundColor Green
    Write-Host "`nYou're ready to run the pipeline!" -ForegroundColor Green
    Write-Host "Execute: .\run_pipeline.ps1" -ForegroundColor White
} elseif ($errors -eq 0) {
    Write-Host "Status: Ready with warnings ($warnings)" -ForegroundColor Yellow
    Write-Host "`nInstall missing packages:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
} else {
    Write-Host "Status: Issues found ($errors errors, $warnings warnings)" -ForegroundColor Red
    Write-Host "`nPlease fix the errors above before proceeding." -ForegroundColor Red
}
Write-Host "========================================" -ForegroundColor Cyan
